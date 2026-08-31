"""
Phase C: SER/STRUT decision-tree transfer refinement (puia/fine_tuning.py).

Refines an ensemble of source-trained sklearn decision trees using data from
a target volcano, following Segev et al. (2016), "Learn on source, refine on
target":

- STRUT: keep each tree's structure (features at nodes) but re-select every
  numeric threshold by maximising class-weighted Gini gain on the target
  samples reaching that node, top-down; leaf probabilities are re-estimated
  from target data.
- SER: expand each original leaf by fitting a small decision tree on the
  target samples reaching it, then reduce bottom-up — collapse any subtree
  whose collapsed-leaf error on target data is no worse than the subtree's.

Simplifications vs the original paper (documented deliberately):
- STRUT uses pure Gini gain on target data (no divergence-to-source term).
- Nodes that no target sample reaches are kept unchanged (fallback to source
  knowledge) rather than pruned.

This module deliberately has no puia imports (numpy/sklearn only) so it can
be used standalone on machines without tsfresh. Import directly, e.g.:
    import importlib.util
    spec = importlib.util.spec_from_file_location('fine_tuning', path)
"""

import numpy as np
from sklearn.tree import DecisionTreeClassifier

MIN_REFIT = 3      # min target samples at a node before STRUT re-fits its threshold
MIN_EXPAND = 4     # min target samples at a leaf before SER expands it
EXPAND_DEPTH = 3   # max depth of SER expansion subtrees


# ============================================================
# Tree extraction / prediction
# ============================================================
def _node_from_sklearn(clf):
    """Convert a fitted sklearn DecisionTreeClassifier into mutable dict nodes."""
    t = clf.tree_
    classes = list(clf.classes_)
    if True in classes:
        pos_idx = classes.index(True)
    else:
        pos_idx = int(np.argmax(classes))

    def build(i):
        left, right = t.children_left[i], t.children_right[i]
        counts = t.value[i][0]
        total = counts.sum()
        proba = float(counts[pos_idx] / total) if total > 0 else 0.
        if left == -1:
            return {'leaf': True, 'proba': proba}
        return {'leaf': False, 'feature': int(t.feature[i]),
                'threshold': float(t.threshold[i]), 'proba': proba,
                'left': build(left), 'right': build(right)}

    return build(0)


class RefinableTree:
    """Mutable decision tree with vectorised prediction."""

    def __init__(self, root):
        self.root = root

    @classmethod
    def from_sklearn(cls, clf):
        return cls(_node_from_sklearn(clf))

    def predict_proba(self, X):
        X = np.asarray(X, dtype=float)
        out = np.empty(len(X))

        def rec(node, idx):
            if len(idx) == 0:
                return
            if node['leaf']:
                out[idx] = node['proba']
                return
            go_left = X[idx, node['feature']] <= node['threshold']
            rec(node['left'], idx[go_left])
            rec(node['right'], idx[~go_left])

        rec(self.root, np.arange(len(X)))
        return out

    def predict(self, X):
        return self.predict_proba(X) >= 0.5

    def n_nodes(self):
        def count(n):
            return 1 if n['leaf'] else 1 + count(n['left']) + count(n['right'])
        return count(self.root)


# ============================================================
# STRUT
# ============================================================
def _best_threshold(v, y, w):
    """Threshold maximising class-weighted Gini gain. Returns (thr, gain)."""
    order = np.argsort(v, kind='mergesort')
    v, y, w = v[order], y[order], w[order]
    W = w.sum()
    Wp = (w * y).sum()
    cw = np.cumsum(w)
    cwp = np.cumsum(w * y)
    valid = np.nonzero(v[:-1] < v[1:])[0]
    if len(valid) == 0:
        return None, 0.
    WL, WPL = cw[valid], cwp[valid]
    WR, WPR = W - WL, Wp - WPL
    pL = WPL / WL
    pR = WPR / np.maximum(WR, 1e-300)
    gL = 1 - pL ** 2 - (1 - pL) ** 2
    gR = 1 - pR ** 2 - (1 - pR) ** 2
    p0 = Wp / W
    g0 = 1 - p0 ** 2 - (1 - p0) ** 2
    gain = g0 - (WL * gL + WR * gR) / W
    k = int(np.argmax(gain))
    thr = (v[valid[k]] + v[valid[k] + 1]) / 2.
    return float(thr), float(gain[k])


def strut(node, X, y, w, min_refit=MIN_REFIT, refit=True, shrink=0.0):
    """Refine thresholds/leaf probabilities in place, top-down.

    refit=False only re-estimates leaf probabilities (calibration-only).
    shrink in [0,1) pulls the refit threshold back toward the source value:
    thr <- shrink*old + (1-shrink)*new (a cheap stand-in for STRUT's
    divergence-to-source regularisation).
    """
    if len(y) == 0:
        return  # no target data reaches here: keep source subtree
    if node['leaf']:
        node['proba'] = float((w * y).sum() / w.sum())
        return
    v = X[:, node['feature']]
    if refit and len(y) >= min_refit and y.min() != y.max():
        thr, gain = _best_threshold(v, y.astype(float), w)
        if thr is not None and gain > 0:
            node['threshold'] = shrink * node['threshold'] + (1 - shrink) * thr
    node['proba'] = float((w * y).sum() / w.sum())
    go_left = v <= node['threshold']
    strut(node['left'], X[go_left], y[go_left], w[go_left], min_refit, refit, shrink)
    strut(node['right'], X[~go_left], y[~go_left], w[~go_left], min_refit, refit, shrink)


# ============================================================
# SER
# ============================================================
def ser(node, X, y, w, expand_depth=EXPAND_DEPTH, min_expand=MIN_EXPAND,
        random_state=0):
    """Expand leaves on target data, then reduce bottom-up. Returns the
    (possibly replaced) node."""
    if node['leaf']:
        if len(y) >= min_expand and y.min() != y.max():
            clf = DecisionTreeClassifier(
                max_depth=expand_depth, class_weight='balanced',
                min_samples_leaf=2, random_state=random_state)
            clf.fit(X, y)
            return _node_from_sklearn(clf)
        if len(y) > 0:
            node['proba'] = float((w * y).sum() / w.sum())
        return node

    v = X[:, node['feature']]
    go_left = v <= node['threshold']
    node['left'] = ser(node['left'], X[go_left], y[go_left], w[go_left],
                       expand_depth, min_expand, random_state)
    node['right'] = ser(node['right'], X[~go_left], y[~go_left], w[~go_left],
                        expand_depth, min_expand, random_state)

    # reduction: collapse subtree if a single leaf does no worse on target data
    if len(y) > 0:
        sub_pred = RefinableTree(node).predict(X)
        werr_sub = (w * (sub_pred != (y > 0))).sum()
        p_leaf = (w * y).sum() / w.sum()
        werr_leaf = (w * ((p_leaf >= 0.5) != (y > 0))).sum()
        if werr_leaf <= werr_sub:
            return {'leaf': True, 'proba': float(p_leaf)}
        node['proba'] = float(p_leaf)
    return node


# ============================================================
# Ensemble helpers
# ============================================================
def build_X(fM, fts):
    """Feature matrix for one tree: its fts columns in order; missing
    features filled with 0, NaN filled with 1e-8 (matches puia forecasting)."""
    X = fM.reindex(columns=fts, fill_value=0.)
    return X.fillna(1.e-8).values.astype(float)


def balanced_weights(y):
    y = np.asarray(y) > 0
    n = len(y)
    npos = max(int(y.sum()), 1)
    nneg = max(n - int(y.sum()), 1)
    return np.where(y, n / (2. * npos), n / (2. * nneg))


def refine_ensemble(trees_fts, fM, y, method, random_state=0,
                    resample_ratio=None, shrink=0.5):
    """Refine a list of (sklearn_clf, fts) with target data.

    Returns list of (RefinableTree, fts).
    method: 'strut' | 'ser' | 'leaf' (calibration-only) | 'strut_shrink'
            (threshold pulled halfway back toward the source value).
    resample_ratio: if set (e.g. 0.75), each tree is refined on its own
    RandomUnderSampler subset (seeded per tree) — mirrors how the source
    trees saw data and preserves ensemble diversity.
    """
    y = np.asarray(y) > 0
    base_rows = np.arange(len(y))
    refined = []
    for i, (clf, fts) in enumerate(trees_fts):
        rows = base_rows
        if resample_ratio is not None:
            from imblearn.under_sampling import RandomUnderSampler
            rus = RandomUnderSampler(sampling_strategy=resample_ratio,
                                     random_state=random_state + i)
            sel, _ = rus.fit_resample(base_rows.reshape(-1, 1), y)
            rows = np.sort(sel[:, 0])
        yi = y[rows]
        wi = balanced_weights(yi)
        X = build_X(fM.iloc[rows], fts)
        rt = RefinableTree.from_sklearn(clf)
        if method in ('strut', 'strut_shrink', 'leaf'):
            strut(rt.root, X, yi.astype(float), wi,
                  refit=(method != 'leaf'),
                  shrink=shrink if method == 'strut_shrink' else 0.0)
        elif method == 'ser':
            rt.root = ser(rt.root, X, yi.astype(float), wi,
                          random_state=random_state)
        else:
            raise ValueError(f"unknown method '{method}'")
        refined.append((rt, fts))
    return refined


def predict_consensus(refined, fM):
    """Mean binary vote of a refined ensemble on feature matrix fM."""
    total = np.zeros(len(fM))
    for rt, fts in refined:
        total += rt.predict(build_X(fM, fts)).astype(float)
    return total / len(refined)


# ============================================================
# Phase D: multi-source TrAdaBoost (Dai et al. 2007)
# ============================================================
class TransferBoostEnsemble:
    """Instance-weighted boosting transfer: source instances that weak
    learners consistently misclassify (w.r.t. target-aware training) are
    progressively downweighted; misclassified TARGET instances are
    upweighted (AdaBoost). Prediction uses the second half of the learners,
    weighted by log(1/beta_t), returning a [0,1] consensus-like score.

    Initial weights are class-balanced within each domain and the two
    domains start with equal total weight (the extreme class imbalance and
    source/target size ratio would otherwise swamp the target).

    Tracks per-source-station weight fractions so we can see which sources
    the boosting decides to keep — the automatic analogue of Phase B's
    pool curation.
    """

    def __init__(self, n_iterations=40, max_depth=6, random_state=0):
        self.n_iterations = n_iterations
        self.max_depth = max_depth
        self.random_state = random_state
        self.learners = []          # (tree, alpha) for the used half
        self.features = None
        self.src_weight_history = []

    def fit(self, Xs, ys, src_station, Xt, yt, features):
        from sklearn.tree import DecisionTreeClassifier
        self.features = list(features)
        Xs = np.asarray(Xs, float)
        Xt = np.asarray(Xt, float)
        ys = np.asarray(ys) > 0
        yt = np.asarray(yt) > 0
        src_station = np.asarray(src_station)
        n_s, n_t = len(ys), len(yt)

        def balanced(y):
            npos = max(int(y.sum()), 1)
            nneg = max(len(y) - int(y.sum()), 1)
            w = np.where(y, 1. / (2 * npos), 1. / (2 * nneg))
            return w / w.sum()

        w_s = balanced(ys)          # sums to 1
        w_t = balanced(yt)          # sums to 1
        beta_src = 1. / (1. + np.sqrt(2. * np.log(max(n_s, 2)) / self.n_iterations))

        X_all = np.vstack([Xs, Xt])
        y_all = np.concatenate([ys, yt])
        all_learners = []
        for t in range(self.n_iterations):
            w = np.concatenate([w_s, w_t])
            w = w / w.sum()
            h = DecisionTreeClassifier(max_depth=self.max_depth,
                                       random_state=self.random_state + t)
            h.fit(X_all, y_all, sample_weight=w)
            pred = h.predict(X_all)
            wrong_s = pred[:n_s] != ys
            wrong_t = pred[n_s:] != yt
            eps = float((w_t / w_t.sum() * wrong_t).sum())
            eps = min(max(eps, 1e-6), 0.499)
            beta_t = eps / (1. - eps)
            all_learners.append((h, np.log(1. / beta_t)))
            # updates: target wrong -> upweight; source wrong -> downweight
            w_t = w_t * np.power(beta_t, -wrong_t.astype(float))
            w_s = w_s * np.power(beta_src, wrong_s.astype(float))
            # renormalise domains jointly (keeps relative source decay)
            z = w_s.sum() + w_t.sum()
            w_s, w_t = w_s / z, w_t / z
            frac = {sta: float(w_s[src_station == sta].sum() / max(w_s.sum(), 1e-300))
                    for sta in np.unique(src_station)}
            frac['_target_total'] = float(w_t.sum() / (w_s.sum() + w_t.sum()))
            self.src_weight_history.append(frac)

        self.learners = all_learners[self.n_iterations // 2:]
        return self

    def predict_score(self, fM):
        X = fM.reindex(columns=self.features, fill_value=0.)
        X = X.fillna(1.e-8).values.astype(float)
        num = np.zeros(len(X))
        den = 0.
        for h, a in self.learners:
            num += a * h.predict(X).astype(float)
            den += a
        return num / max(den, 1e-300)
