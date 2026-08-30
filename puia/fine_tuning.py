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


def strut(node, X, y, w, min_refit=MIN_REFIT):
    """Refine thresholds/leaf probabilities in place, top-down."""
    if len(y) == 0:
        return  # no target data reaches here: keep source subtree
    if node['leaf']:
        node['proba'] = float((w * y).sum() / w.sum())
        return
    v = X[:, node['feature']]
    if len(y) >= min_refit and y.min() != y.max():
        thr, gain = _best_threshold(v, y.astype(float), w)
        if thr is not None and gain > 0:
            node['threshold'] = thr
    node['proba'] = float((w * y).sum() / w.sum())
    go_left = v <= node['threshold']
    strut(node['left'], X[go_left], y[go_left], w[go_left], min_refit)
    strut(node['right'], X[~go_left], y[~go_left], w[~go_left], min_refit)


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


def refine_ensemble(trees_fts, fM, y, method, random_state=0):
    """Refine a list of (sklearn_clf, fts) with target data.

    Returns list of (RefinableTree, fts). method: 'strut' | 'ser'.
    """
    y = np.asarray(y) > 0
    w = balanced_weights(y)
    refined = []
    for clf, fts in trees_fts:
        X = build_X(fM, fts)
        rt = RefinableTree.from_sklearn(clf)
        if method == 'strut':
            strut(rt.root, X, y.astype(float), w)
        elif method == 'ser':
            rt.root = ser(rt.root, X, y.astype(float), w,
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
