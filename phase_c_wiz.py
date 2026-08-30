"""
Phase C for WIZ (Whakaari) as target: SER/STRUT refinement of the WIZ__full
base ensemble ({FWVZ,KRVZ,ONTA,SHW}) using WIZ's own data.

WIZ is the best-instrumented target (5 eruptions, 20 positive training
windows vs 8-12 for KRVZ/FWVZ), so this also tests whether Phase C's
overfitting failure mode relaxes with more target eruptions.

Runs all four methods: plain strut/ser (round-1 style) and the per-tree
undersampled strut_us/ser_us (round-2 winners). Requires phase_b_wiz.py to
have completed (WIZ__full models + consensus master).

Thin driver: overrides phase_c_local's module config, then runs its stages.
Usage:  python -u phase_c_wiz.py
"""

import phase_c_local as PC

PC.TARGETS = ['WIZ']
PC.BASES = ['full']
PC.METHODS = ['strut', 'ser', 'strut_us', 'ser_us']
PC.RESULTS_CSV = 'phase_c_results_WIZ.csv'
PC.data = dict(PC.data)  # WIZ window already correct in config

if __name__ == '__main__':
    PC.log.info('PHASE C — WIZ target driver')
    PC.stage_refine()
    PC.stage_forecast()
    PC.stage_eval()
    PC.log.info('PHASE C (WIZ) done.')
