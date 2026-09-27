"""CMTF diagnostic experiment, created 2026-09-06.

Synthetic evidence only. No market data, LLM, fitted HMM, or trade execution.
Training regimes and nominal filtering parameters are deliberately oracle inputs.
Run: python3 synthetic_study.py (outputs beside this script).
"""
from pathlib import Path
import json
import platform
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
REPS, NTRAIN, NTEST = 200, 2000, 1000
A = np.array([[.97, .03], [.03, .97]])
MEANS = np.array([-.7, .7])
SIGMA = 1.3

def generate(rng, theta, shift=False):
    n = NTRAIN + NTEST
    state = np.zeros(n, dtype=int)
    state[0] = rng.integers(2)
    for t in range(1, n):
        state[t] = rng.choice(2, p=A[state[t-1]])
    c = rng.normal(size=n)
    x = c + rng.normal(size=n)
    y = theta[state]*x + 2*c + rng.normal(size=n)
    m = MEANS[state] + SIGMA*rng.normal(size=n)
    if shift:
        m[NTRAIN:] = -MEANS[state[NTRAIN:]] + SIGMA*rng.normal(size=NTEST)
    return state, c, x, y, m

def filter_states(m):
    q = np.array([.5, .5])
    out = np.zeros((len(m), 2))
    for t, obs in enumerate(m):
        pred = q if t == 0 else q @ A
        log_lik = -.5*((obs-MEANS)/SIGMA)**2
        lik = np.exp(log_lik-log_lik.max())
        q = pred*lik
        q /= q.sum()
        out[t] = q
    return out

def slope(x, y, c, adjust):
    design = np.column_stack([np.ones(len(x)), x, c]) if adjust else np.column_stack([np.ones(len(x)), x])
    return np.linalg.lstsq(design, y, rcond=None)[0][1]

def main():
    (ROOT / "assets").mkdir(parents=True, exist_ok=True)
    records, diagnostic = [], []
    scenarios = [('Observed confounder', np.array([1.,-1.]), True, False),
                 ('Hidden confounder', np.array([1.,-1.]), False, False),
                 ('Emission shift', np.array([1.,-1.]), True, True),
                 ('Inactive second regime', np.array([1.,0.]), True, False)]
    for case, theta, adjust, shift in scenarios:
        for rep in range(REPS):
            rng = np.random.default_rng(20260906 + rep)
            state, c, x, y, m = generate(rng, theta, shift)
            q = filter_states(m)[NTRAIN:]
            train, test = slice(0,NTRAIN), slice(NTRAIN,None)
            beta = np.array([slope(x[:NTRAIN][state[:NTRAIN]==k], y[:NTRAIN][state[:NTRAIN]==k], c[:NTRAIN][state[:NTRAIN]==k], adjust) for k in range(2)])
            preds = {
                'Pooled unadjusted': np.full(NTEST, slope(x[train],y[train],c[train],False)),
                'Pooled available adjustment': np.full(NTEST, slope(x[train],y[train],c[train],adjust)),
                'Filtered hard gate': beta[np.argmax(q,axis=1)],
                'Filtered soft gate': q @ beta,
                'Oracle state gate': beta[state[test]],
            }
            truth = theta[state[test]]
            for model, pred in preds.items():
                records.append({'scenario':case,'replicate':rep,'method':model,
                                'mse':float(np.mean((pred-truth)**2)),
                                'mae':float(np.mean(np.abs(pred-truth)))})
            diagnostic.append({'scenario':case,'replicate':rep,
                               'state_accuracy':float(np.mean(q.argmax(axis=1)==state[test])),
                               'beta0':float(beta[0]),'beta1':float(beta[1])})
    raw = pd.DataFrame(records)
    raw.to_csv(ROOT/'synthetic_replicates.csv',index=False)
    summary = raw.groupby(['scenario','method'],sort=False).agg(
        mean_mse=('mse','mean'),sd_mse=('mse','std'),
        mean_mae=('mae','mean'),replicates=('mse','size')).reset_index()
    summary['mcse_mse'] = summary.sd_mse/np.sqrt(REPS)
    summary.to_csv(ROOT/'synthetic_summary.csv',index=False)
    pd.DataFrame(diagnostic).to_csv(ROOT/'synthetic_diagnostics.csv',index=False)
    paired=[]
    for case in raw.scenario.unique():
        p=raw[raw.scenario==case].pivot(index='replicate',columns='method',values='mse')
        d=p['Filtered hard gate']-p['Filtered soft gate']
        paired.append({'scenario':case,'hard_minus_soft_mse':float(d.mean()),'paired_mcse':float(d.std()/np.sqrt(REPS))})
    (ROOT/'synthetic_paired.json').write_text(json.dumps(paired,indent=2))
    (ROOT/'experiment_manifest.json').write_text(json.dumps({
        'seed_rule':'20260906 + replicate, replicate=0,...,199',
        'replicates':REPS,'training_length':NTRAIN,'test_length':NTEST,
        'transition':A.tolist(),'emission_means':MEANS.tolist(),'emission_sd':SIGMA,
        'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__,
        'training_states':'oracle observed','filter_parameters':'oracle nominal, frozen',
        'test_filter_inputs':'M only; no outcomes, treatment, confounder, or true state',
        'outcome':'synthetic structural coefficient MSE, not return'},indent=2))
    colors=['#adb8c4','#697d91','#d78845','#167a89','#355636']
    fig, axs=plt.subplots(2,2,figsize=(10,7.5),layout='constrained')
    for ax,(case,_,_,_) in zip(axs.flat,scenarios):
        s=summary[summary.scenario==case]
        ax.barh(range(5),s.mean_mse,color=colors,xerr=1.96*s.mcse_mse,capsize=3)
        ax.set_yticks(range(5),['Pooled naive','Pooled adjusted*','Hard gate','Soft gate','Oracle state'])
        ax.invert_yaxis(); ax.set_title(case,loc='left',fontweight='bold')
        ax.set_xlabel('Structural coefficient MSE'); ax.spines[['top','right']].set_visible(False)
    fig.suptitle('Synthetic diagnostic: regime gating helps only under its assumptions',fontsize=13)
    fig.savefig(ROOT/'assets/synthetic_results.png',dpi=180)
    plt.close(fig)
    print(summary.to_string(index=False))
    print(json.dumps(paired,indent=2))

if __name__ == '__main__':
    main()
