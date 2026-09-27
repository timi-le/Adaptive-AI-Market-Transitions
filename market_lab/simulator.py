from dataclasses import dataclass, asdict
from itertools import product
import numpy as np
import pandas as pd

@dataclass(frozen=True)
class Config:
    steps: int = 200
    agents: int = 40
    capital: float = 1_000_000.0
    adaptive_share: float = 0.5
    depth: float = 200_000.0  # currency notional / unit log impact
    trade_fraction: float = 0.02  # per-agent per-step notional / positive equity
    max_volume: float = 0.25  # aggregate gross executed notional / depth per step
    impact: float = 0.03
    spread_bps: float = 4.0
    fee_bps: float = 1.0
    fundamental_vol: float = 0.002
    price_noise: float = 0.0005
    fundamental_pull: float = 0.05
    shock_step: int = 70
    shock_size: float = -0.08
    shared_error: float = 0.5
    perturbation_sd: float = 0.20
    portfolio_noise_sd: float = 0.10
    adaptation_rate: float = 0.02
    adaptation_interval: int = 5
    gate_threshold: float = 0.15
    drawdown_aversion: float = 3.0
    crash_drawdown: float = 0.10
    initial_price: float = 100.0
    value_feedback: float = 1.0
    momentum_feedback: float = 15.0
    target_leverage: float = 1.0  # target multiplier, not a broker margin model
    impact_function: str = 'tanh'

    def validate(self):
        for key,value in asdict(self).items():
            if isinstance(value,(int,float)) and not np.isfinite(value): raise ValueError(f'Nonfinite {key}')
        if self.impact_function not in ('tanh','linear'): raise ValueError('Unknown impact function')
        if min(self.value_feedback,self.momentum_feedback,self.target_leverage)<0: raise ValueError('Negative feedback/leverage')
        if self.steps < 3 or self.agents < 2: raise ValueError('Need >=3 steps and >=2 agents')
        if min(self.capital,self.depth,self.initial_price)<=0: raise ValueError('Positive capital/depth/price required')
        for x in [self.adaptive_share,self.shared_error,self.trade_fraction,self.max_volume]:
            if not 0<=x<=1: raise ValueError('Fractions must lie in [0,1]')
        if self.adaptation_interval<1 or not 0<=self.shock_step<self.steps: raise ValueError('Invalid clock')
        if min(self.impact,self.spread_bps,self.fee_bps,self.fundamental_vol,self.price_noise,self.fundamental_pull,self.perturbation_sd,self.portfolio_noise_sd,self.adaptation_rate,self.gate_threshold,self.drawdown_aversion,self.crash_drawdown)<0: raise ValueError('Negative parameter')


def simulate(config=Config(), seed=0, shock=False, reference=None, kernels=None):
    """One asset, aggregate impact, external dealer. No LLM/API/broker is invoked.

    reference: steps rows with signal [-1,1], risk_budget [0,1], gate [0,1].
    kernels: optional trade/portfolio 3x3 conditional substitution probabilities.
    Return episode dataframe, measured summaries, final accounting diagnostics.
    """
    c=config;c.validate();n=c.agents
    market_rng=np.random.default_rng(np.random.SeedSequence([seed,100]))
    eps=market_rng.normal(size=(c.steps,2))
    common=np.random.default_rng(np.random.SeedSequence([seed,200])).normal(size=c.steps)
    noise=np.column_stack([np.random.default_rng(np.random.SeedSequence([seed,300,i])).normal(size=c.steps) for i in range(n)])
    uniforms=np.stack([np.random.default_rng(np.random.SeedSequence([seed,400,i])).random((c.steps,2)) for i in range(n)],axis=1)
    ai=np.arange(n)<max(1,n//2);na=int(ai.sum());nb=n-na
    initial=np.where(ai,c.capital*c.adaptive_share/na,c.capital*(1-c.adaptive_share)/nb)
    cash=initial.copy();inventory=np.zeros(n);peak=initial.copy();gain=np.ones(n)
    dealer_cash=0.;dealer_inventory=0.;fee_account=0.;price=c.initial_price;fundamental=price;last_return=0.
    price_peak=price;previous_equity=initial.copy();previous_signal=np.zeros(n);rows=[]
    if reference is not None:
        if len(reference)<c.steps: raise ValueError('Reference shorter than simulation; no automatic repeats')
        required=['signal','risk_budget','gate']
        a=reference[required].to_numpy(float)
        if not np.isfinite(a).all() or (np.abs(a[:,0])>1).any() or ((a[:,1:]<0)|(a[:,1:]>1)).any():raise ValueError('Invalid reference values')
    if kernels is not None:
        for key in ['trade','portfolio']:
            k=np.asarray(kernels[key],float)
            if k.shape!=(3,3) or (k<0).any() or not np.allclose(k.sum(1),1):raise ValueError('Invalid kernel')
    for t in range(c.steps):
        equity=cash+inventory*price;peak=np.maximum(peak,equity)
        dd=np.maximum(0,1-equity/np.maximum(peak,1e-12))
        alive=equity>0
        if t and t%c.adaptation_interval==0:
            performance=np.clip((equity-previous_equity)/np.maximum(np.abs(previous_equity),1),-.1,.1)
            gain=np.clip(gain+c.adaptation_rate*performance,.5,1.5)
        previous_equity=equity.copy()
        # Agent observes current state only. Exogenous innovations below occur after orders.
        value_signal=np.tanh(8*np.log(fundamental/price))
        replay_signal=0. if reference is None else float(reference.iloc[t]['signal'])
        base_budget=.6 if reference is None else float(reference.iloc[t]['risk_budget'])
        recorded_gate=1. if reference is None else float(reference.iloc[t]['gate'])
        base_trade=np.full(n,replay_signal)
        budgets=np.full(n,base_budget)
        if kernels is not None:
            ti=int(np.argmin(abs(np.array([-1,0,1])-replay_signal)))
            pi=int(np.argmin(abs(np.array([0,.5,1])-base_budget)))
            base_trade[ai]=np.array([-1,0,1])[np.searchsorted(np.cumsum(kernels['trade'][ti]),uniforms[t,ai,0],side='right').clip(0,2)]
            budgets[ai]=np.array([0,.5,1])[np.searchsorted(np.cumsum(kernels['portfolio'][pi]),uniforms[t,ai,1],side='right').clip(0,2)]
        error=c.perturbation_sd*(np.sqrt(c.shared_error)*common[t]+np.sqrt(1-c.shared_error)*noise[t])
        adaptive=np.clip(gain*(base_trade+c.value_feedback*value_signal+c.momentum_feedback*last_return)+error,-1,1)
        conventional=np.clip(value_signal+.05*noise[t],-1,1)
        signal=np.where(ai,adaptive,conventional)
        # A threshold gate is only a proxy ablation, not a validated CMTF evidence gate.
        gate=np.where(ai,(np.abs(signal)>=c.gate_threshold)*recorded_gate,1.)
        budgets=np.clip(budgets+c.portfolio_noise_sd*noise[t],0,1)
        budget=np.where(ai,budgets*np.exp(-c.drawdown_aversion*dd),.5)
        target=signal*budget*c.target_leverage*np.maximum(equity,0)/price
        # Gate freezes the speculative target; drawdown/bankruptcy exits stay separate.
        target=np.where(gate>0,target,inventory)
        forced=(~alive)|(dd>.5)
        target=np.where(forced,0,target)
        cap=c.trade_fraction*np.maximum(equity,0)/price
        cap=np.where(forced,np.abs(inventory),cap)
        orders=np.clip(target-inventory,-cap,cap)
        gross_notional=float(np.abs(orders).sum()*price)
        participation=min(1.,c.max_volume*c.depth/max(gross_notional,1e-12))
        quantities=orders*participation
        net_notional=float(quantities.sum()*price)
        # Same exogenous streams across all parameter cells sharing the seed.
        innovation=c.fundamental_vol*eps[t,0]+(c.shock_size if shock and t==c.shock_step else 0)
        next_fundamental=fundamental*np.exp(innovation)
        scaled_flow=net_notional/c.depth
        impact_response=np.tanh(scaled_flow) if c.impact_function=='tanh' else scaled_flow
        log_move=c.fundamental_pull*np.log(next_fundamental/price)+c.impact*impact_response+c.price_noise*eps[t,1]
        next_price=price*np.exp(log_move)
        execution=np.sqrt(price*next_price)*np.exp(np.sign(quantities)*c.spread_bps/20000)
        fees=np.abs(quantities)*execution*c.fee_bps/10000
        cash-=quantities*execution+fees;inventory+=quantities
        dealer_cash+=float((quantities*execution).sum());dealer_inventory-=float(quantities.sum());fee_account+=float(fees.sum())
        new_equity=cash+inventory*next_price
        active_ai=ai & (initial>0)
        labels=np.where(gate[active_ai]>0,np.sign(signal[active_ai]),0)
        counts=np.unique(labels,return_counts=True)[1]
        agreement=float(np.sum(counts*(counts-1))/(len(labels)*(len(labels)-1))) if len(labels)>1 else np.nan
        price_peak=max(price_peak,next_price)
        rows.append(dict(step=t,price=next_price,fundamental=next_fundamental,log_return=log_move,
          ai_direction_agreement=agreement,price_drawdown=1-next_price/price_peak,ai_equity=float(new_equity[ai].sum()),background_equity=float(new_equity[~ai].sum()),
          executed_gross_notional=float((np.abs(quantities)*execution).sum()),net_order_notional=net_notional,
          fill_fraction=participation,ai_action_coverage=float((gate[ai]>0).mean()),fees_cumulative=fee_account,
          cash_residual=float(cash.sum()+dealer_cash+fee_account-c.capital),inventory_residual=float(inventory.sum()+dealer_inventory),
          bankrupt_agents=int((new_equity<0).sum()),mean_gain=float(gain[ai].mean())))
        previous_signal=signal.copy();last_return=float(log_move);price=next_price;fundamental=next_fundamental
    frame=pd.DataFrame(rows);den=float(initial[ai].sum());mdd=float(frame.price_drawdown.max())
    # No annualization or alpha claim: step duration and a factor benchmark are unspecified.
    summary=dict(seed=seed,shock=shock,**asdict(c),mode='synthetic_proxy' if reference is None else 'historical_output_replay_plus_feedback_proxy',
       ai_net_return=None if den==0 else float(frame.ai_equity.iloc[-1]/den-1),
       max_price_drawdown=mdd,crash=int(mdd>=c.crash_drawdown),return_volatility=float(frame.log_return.std(ddof=1)),
       turnover=float(frame.executed_gross_notional.sum()/c.capital),mean_fill_fraction=float(frame.fill_fraction.mean()),
       max_bankrupt_agents=int(frame.bankrupt_agents.max()),max_abs_cash_residual=float(frame.cash_residual.abs().max()),
       max_abs_inventory_residual=float(frame.inventory_residual.abs().max()))
    summary['mean_policy_agreement']=float(frame.ai_direction_agreement.mean())
    summary['mean_action_coverage']=float(frame.ai_action_coverage.mean())
    summary['price_return']=float(price/c.initial_price-1)
    return frame,summary


def paired_grid(base, factors, seeds=(0,1,2), reference=None, kernels=None):
    """Full factorial; each cell/seed is run both with and without the shock."""
    records=[]
    for values in product(*factors.values()):
        params={**asdict(base),**dict(zip(factors,values))}
        for seed in seeds:
            for shock in [False,True]:
                _,s=simulate(Config(**params),int(seed),shock,reference,kernels);records.append(s)
    return pd.DataFrame(records)
