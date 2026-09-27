import unittest,tempfile
from pathlib import Path
from dataclasses import replace
import numpy as np
import pandas as pd
from market_lab import Config,simulate
from market_lab.logs import load_reference,fit_kernels

class LabTests(unittest.TestCase):
    def test_repeatability_and_accounting(self):
        c=Config(steps=80,shock_step=30)
        a,s=simulate(c,12,True);b,_=simulate(c,12,True)
        pd.testing.assert_frame_equal(a,b)
        self.assertLess(s['max_abs_cash_residual'],1e-7)
        self.assertLess(s['max_abs_inventory_residual'],1e-9)
    def test_common_exogenous_path(self):
        c=Config(steps=80,shock_step=30)
        a,_=simulate(c,8,True);b,_=simulate(replace(c,agents=21,depth=5e5),8,True)
        np.testing.assert_array_equal(a.fundamental,b.fundamental)
    def test_no_trading_preserves_agent_cash(self):
        c=Config(steps=80,shock_step=30,trade_fraction=0)
        a,s=simulate(c,3)
        self.assertEqual(s['turnover'],0);self.assertAlmostEqual(s['ai_net_return'],0)
    def test_capital_depth_scale_invariance(self):
        c=Config(steps=80,shock_step=30)
        a,s=simulate(c,4);b,t=simulate(replace(c,capital=c.capital*10,depth=c.depth*10),4)
        np.testing.assert_allclose(a.price,b.price,rtol=1e-12)
        self.assertAlmostEqual(s['ai_net_return'],t['ai_net_return'])
    def test_future_unavailable_log_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'events.csv'
            pd.DataFrame([dict(role='trade',symbol='DEMO',model='t',decision_time='2026-01-01T00:00:00Z',available_at='2026-01-01T00:01:00Z',value=1,source_id='x')]).to_csv(p,index=False)
            with self.assertRaises(ValueError):load_reference(p,['2026-01-01T00:00:00Z'],'DEMO')
    def test_missing_and_stale_outputs_gate_off(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'events.csv'
            rows=[dict(role=r,symbol='DEMO',model=r,decision_time='2026-01-01T00:01:00Z',available_at='2026-01-01T00:01:00Z',value=.5,source_id=r) for r in ['trade','portfolio']]
            pd.DataFrame(rows).to_csv(p,index=False)
            x=load_reference(p,['2026-01-01T00:00:00Z','2026-01-01T00:01:00Z','2026-01-01T02:00:00Z'],'DEMO',max_age='1h')
            self.assertEqual(x.gate.tolist(),[0,1,0])
    def test_unmatched_benchmark_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'cases.csv'
            pd.DataFrame([dict(role='trade',case_id=1,reference_input_hash='a',candidate_input_hash='b',reference_action=0,candidate_action=1)]).to_csv(p,index=False)
            with self.assertRaises(ValueError):fit_kernels(p,min_cases=1)
    def test_valid_matched_kernel(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'cases.csv'
            pd.DataFrame([dict(role=r,case_id=j,reference_input_hash=str(j),candidate_input_hash=str(j),reference_action=j%3,candidate_action=j%3) for r in ['trade','portfolio'] for j in range(30)]).to_csv(p,index=False)
            k=fit_kernels(p)
            np.testing.assert_allclose(np.array(k['trade']).sum(1),1)
            self.assertGreater(k['trade'][0][0],k['trade'][0][1])

if __name__=='__main__':unittest.main()
