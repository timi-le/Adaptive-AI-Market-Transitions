"""Regenerate synthetic findings, then verify the headline market contrast table.
No model APIs, broker connections, private logs, or trading actions are used.
"""
from pathlib import Path
import shutil
import pandas as pd
import numpy as np
from scripts.experiments import run_market_grid,run_profiles,run_robustness,participation_contrasts,interval
from cmtf import synthetic_study
ROOT=Path(__file__).resolve().parent

def main():
    # Keep checked-in values as references before regenerating each experiment.
    expected=pd.read_csv(ROOT/'results/participation_contrast_summary.csv')
    run_market_grid(ROOT)
    profiles,_=run_profiles(ROOT)
    run_robustness(ROOT)
    contrasts=participation_contrasts(profiles)
    contrasts.to_csv(ROOT/'results/participation_contrasts.csv',index=False)
    rows=[]
    for (profile,shared),group in contrasts.groupby(['profile','shared_error']):
        for metric in ['participation_effect_no_shock','participation_shock_interaction']:
            rows.append(dict(profile=profile,shared_error=shared,metric=metric,**interval(group[metric])))
    actual=pd.DataFrame(rows)
    keys=['profile','shared_error','metric']
    left=actual.sort_values(keys).reset_index(drop=True)
    right=expected.sort_values(keys).reset_index(drop=True)
    assert left[keys].equals(right[keys]), 'Contrast identifiers changed'
    np.testing.assert_allclose(left[['mean','se','low95','high95']],right[['mean','se','low95','high95']],rtol=1e-9,atol=1e-11)
    actual.to_csv(ROOT/'results/participation_contrast_summary.csv',index=False)
    synthetic_study.main()
    for source in (ROOT/'cmtf').glob('synthetic_*.*'):
        if source.suffix in ['.csv','.json']:shutil.copy(source,ROOT/'results/cmtf'/source.name)
    for name in ['market_grid_episodes.csv','profile_episodes.csv','robustness_episodes.csv']:
        data=pd.read_csv(ROOT/'results'/name)
        assert data.max_abs_cash_residual.max()<1e-6
        assert data.max_abs_inventory_residual.max()<1e-8
    from scripts.make_figures import main as make_figures
    make_figures()
    print('PASS: 1,312 market episodes, reported contrasts, and synthetic CMTF diagnostic regenerated.')
if __name__=='__main__':main()
