"""Draw manuscript figures from released aggregate result tables."""
from pathlib import Path
import json,shutil
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
def main():
    assets=ROOT/'papers/market/assets'
    if not assets.exists():return  # anonymous code-only supplement has no manuscript tree
    report=json.loads((ROOT/'results/common_panel.json').read_text())
    names=['claude-sonnet-4-6','gpt-4.1-mini','llama-3.3-70b','mistral-large-3']
    labels=['Claude','GPT','Llama','Mistral'];a=np.eye(4)*100
    for pair in report['common_panel_pairwise']:
        i=names.index(pair['model_a']);j=names.index(pair['model_b']);a[i,j]=a[j,i]=pair['raw_agreement_pct']
    fig,ax=plt.subplots(figsize=(4.8,4.1),layout='constrained')
    im=ax.imshow(a,cmap='Blues',vmin=0,vmax=100)
    ax.set_xticks(range(4),labels);ax.set_yticks(range(4),labels)
    for i in range(4):
        for j in range(4):ax.text(j,i,f'{a[i,j]:.1f}%',ha='center',va='center',color='white' if a[i,j]>75 else 'black')
    ax.set_title('Agreement on 5,998 common exported inputs',fontsize=10)
    fig.colorbar(im,ax=ax,label='Directional agreement (%)',shrink=.8)
    fig.savefig(assets/'common_panel_agreement.pdf');plt.close(fig)
    df=pd.read_csv(ROOT/'results/participation_contrast_summary.csv')
    d=df[(df.shared_error==.9)&(df.metric=='participation_shock_interaction')].sort_values('profile')
    mean=d['mean'].to_numpy()*100;low=d.low95.to_numpy()*100;high=d.high95.to_numpy()*100
    fig,ax=plt.subplots(figsize=(5.2,3.5),layout='constrained')
    ax.errorbar(range(6),mean,yerr=np.vstack([mean-low,high-mean]),fmt='o',capsize=4,color='#167a89')
    ax.axhline(0,color='gray',lw=.8);ax.set_xticks(range(6),[f'P{i}' for i in range(6)])
    ax.set_xlabel('Assumed response profile');ax.set_ylabel('Drawdown interaction (percentage points)')
    ax.set_title('Participation × shock; shared-error setting 0.9',fontsize=10)
    fig.savefig(assets/'profile_interactions.pdf');plt.close(fig)
    source=ROOT/'cmtf/assets/synthetic_results.png'
    if source.exists():shutil.copy(source,ROOT/'papers/cmtf/assets/synthetic_results.png')
if __name__=='__main__':main()
