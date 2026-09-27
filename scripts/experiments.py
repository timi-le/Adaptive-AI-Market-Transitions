"""Predeclared mechanistic sensitivity experiments; no LLM performance claims."""
from pathlib import Path
from dataclasses import replace
import json
import numpy as np
import pandas as pd
from scipy.stats import t as student_t
from market_lab import Config,simulate,paired_grid

# These six profiles are assumptions, not benchmark-fitted named models.
PROFILES={
 'P0_reference_proxy':dict(perturbation_sd=0.,portfolio_noise_sd=0.,shared_error=0.,adaptation_rate=0.),
 'P1_low_noise':dict(perturbation_sd=.05,portfolio_noise_sd=.02,shared_error=0.),
 'P2_high_noise':dict(perturbation_sd=.30,portfolio_noise_sd=.15,shared_error=0.),
 'P3_stronger_feedback':dict(perturbation_sd=.30,portfolio_noise_sd=.15,momentum_feedback=30.),
 'P4_slow_adaptation':dict(perturbation_sd=.15,portfolio_noise_sd=.05,shared_error=.5,adaptation_interval=20),
 'P5_selective_proxy_gate':dict(perturbation_sd=.15,portfolio_noise_sd=.05,shared_error=.5,gate_threshold=.40,drawdown_aversion=6.),
}

def paired_shock_table(frame,keys):
    idx=list(keys)+['seed'];metrics=['max_price_drawdown','crash','ai_net_return','turnover']
    if frame.duplicated(idx+['shock']).any():raise ValueError('Duplicate experimental cell')
    off=frame[~frame.shock].set_index(idx)[metrics].sort_index()
    on=frame[frame.shock].set_index(idx)[metrics].sort_index()
    if not off.index.equals(on.index):raise ValueError('Unpaired shock scenarios')
    return (on-off).add_prefix('shock_effect_').reset_index()

def interval(values):
    a=np.asarray(values,dtype=float);a=a[np.isfinite(a)];n=len(a)
    if n<2:return {'n':n,'mean':float(np.mean(a)) if n else None,'se':None,'low95':None,'high95':None}
    mu=float(a.mean());se=float(a.std(ddof=1)/np.sqrt(n));w=float(student_t.ppf(.975,n-1))*se
    return {'n':n,'mean':mu,'se':se,'low95':mu-w,'high95':mu+w}

def summarize_paired(frame,keys):
    p=paired_shock_table(frame,keys);out=[]
    for key,g in p.groupby(keys,dropna=False):
        if not isinstance(key,tuple):key=(key,)
        for metric in [c for c in p if c.startswith('shock_effect_')]:
            out.append({**dict(zip(keys,key)),'metric':metric,**interval(g[metric])})
    return p,pd.DataFrame(out)

def run_market_grid(root):
    root=Path(root);spec=json.loads((root/'configs/demo.json').read_text())
    base=Config(**spec['base']);factors=spec['factors'];episodes=paired_grid(base,factors,spec['seeds'])
    episodes.to_csv(root/'results/market_grid_episodes.csv',index=False)
    paired,summary=summarize_paired(episodes,list(factors))
    paired.to_csv(root/'results/market_grid_paired.csv',index=False)
    summary.to_csv(root/'results/market_grid_intervals.csv',index=False)
    return episodes,summary

def run_profiles(root,reference=None):
    root=Path(root);spec=json.loads((root/'configs/profile_study.json').read_text())
    base=Config(**spec['base']);frames=[]
    for name,settings in PROFILES.items():
        frame=paired_grid(replace(base,**settings),spec['factors'],spec['seeds'],reference=reference)
        frame['profile']=name;frames.append(frame)
    df=pd.concat(frames,ignore_index=True)
    keys=['profile']+list(spec['factors']);p,s=summarize_paired(df,keys)
    df.to_csv(root/'results/profile_episodes.csv',index=False)
    p.to_csv(root/'results/profile_paired.csv',index=False)
    s.to_csv(root/'results/profile_intervals.csv',index=False)
    (root/'results/profile_parameters.json').write_text(json.dumps({'assumed_profiles':PROFILES,'study':spec,'interpretation':'Within-simulator sensitivities. Intervals quantify Monte Carlo variability only; no market/model calibration uncertainty.'},indent=2))
    return df,s

def run_robustness(root):
    root=Path(root);base=Config(steps=120,shock_step=40)
    factors={'impact_function':['tanh','linear'],'target_leverage':[1.,2.],'adaptation_interval':[1,20]}
    df=paired_grid(base,factors,seeds=range(10))
    df.to_csv(root/'results/robustness_episodes.csv',index=False)
    return df

def participation_contrasts(df):
    keys=['profile','shared_error','seed']
    p=df.pivot(index=keys,columns=['adaptive_share','shock'],values='max_price_drawdown')
    return pd.DataFrame({'participation_effect_no_shock':p[(.75,False)]-p[(.25,False)],'participation_shock_interaction':(p[(.75,True)]-p[(.75,False)])-(p[(.25,True)]-p[(.25,False)])}).reset_index()
