"""Strict normalized one-asset log interface. Does not guess production schemas."""
import json
import numpy as np
import pandas as pd

REQUIRED=['role','symbol','model','decision_time','available_at','value','source_id']

def load_reference(path,times,symbol,max_age='1h'):
    events=pd.read_csv(path)
    if set(REQUIRED)-set(events):raise ValueError('Missing columns: '+str(set(REQUIRED)-set(events)))
    if not events.role.isin(['trade','portfolio']).all():raise ValueError('Unknown role')
    if events[REQUIRED].isna().any().any():raise ValueError('Null required field')
    for col in ['decision_time','available_at']:events[col]=pd.to_datetime(events[col],utc=True,errors='raise')
    # decision_time is when action could actually be used, not the original request time.
    if (events.available_at>events.decision_time).any():raise ValueError('Output not available at decision time')
    events=events[events.symbol==symbol].copy()
    if events.empty:raise ValueError('No events for symbol')
    if events.duplicated(['role','decision_time']).any():raise ValueError('Ambiguous simultaneous role outputs')
    trade=events.role=='trade';v=pd.to_numeric(events.value,errors='raise')
    if not np.isfinite(v).all() or (v[trade].abs()>1).any() or ((v[~trade]<0)|(v[~trade]>1)).any():raise ValueError('Invalid values')
    if events.groupby('role').model.nunique().max()>1:raise ValueError('Choose one reference model per role before import')
    clock=pd.DataFrame({'time':pd.to_datetime(times,utc=True)}).sort_values('time')
    if clock.time.duplicated().any():raise ValueError('Duplicate simulation times')
    result=clock.copy();fresh=np.ones(len(clock),bool)
    for role,out in [('trade','signal'),('portfolio','risk_budget')]:
        part=events[events.role==role].sort_values('decision_time')
        if part.empty:raise ValueError('Both trade and portfolio roles are required')
        joined=pd.merge_asof(clock,part[['decision_time','value']],left_on='time',right_on='decision_time',direction='backward',tolerance=pd.Timedelta(max_age))
        fresh &= joined.value.notna().to_numpy()
        result[out]=joined.value.fillna(0).to_numpy()
    result['gate']=fresh.astype(float)
    return result


def fit_kernels(path,min_cases=30,smoothing=1.0):
    """Matched-input conditional substitution, not benchmark accuracy calibration.

    CSV: role,case_id,reference_input_hash,candidate_input_hash,reference_action,candidate_action.
    Both actions are integers 0..2: trade SHORT/NEUTRAL/LONG; portfolio budgets 0/.5/1.
    One candidate/reference pair per file; provenance supplied separately.
    """
    x=pd.read_csv(path);needed=['role','case_id','reference_input_hash','candidate_input_hash','reference_action','candidate_action']
    if set(needed)-set(x):raise ValueError('Missing matched-case columns')
    if x[needed].isna().any().any() or not x.role.isin(['trade','portfolio']).all():raise ValueError('Invalid role or missing value')
    if (x.reference_input_hash!=x.candidate_input_hash).any():raise ValueError('Inputs are not matched')
    if x.duplicated(['role','case_id']).any():raise ValueError('Repeated calibration cases')
    if smoothing<0:raise ValueError('Negative smoothing')
    result={}
    for role in ['trade','portfolio']:
        a=x[x.role==role]
        if len(a)<min_cases:raise ValueError('Too few matched cases for '+role)
        counts=np.full((3,3),float(smoothing))
        for r,c in zip(a.reference_action,a.candidate_action):
            if r not in [0,1,2] or c not in [0,1,2]:raise ValueError('Actions must be 0,1,2')
            counts[int(r),int(c)]+=1
        if (counts.sum(1)==0).any():raise ValueError('Unobserved reference action without smoothing')
        result[role]=(counts/counts.sum(1,keepdims=True)).tolist()
    return result
