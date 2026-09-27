import unittest
import sys
from pathlib import Path
from dataclasses import replace
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "cmtf"))
from evidence_router import Evidence, route, example

class RouterContract(unittest.TestCase):
    def setUp(self):
        self.r = Evidence("a", 1, 0, 0, 0, 90, 110, True, "fixture")
    def test_negative_response_favors_short(self):
        out = example()
        self.assertGreater(out["weights"]["short"], out["weights"]["long"])
        self.assertAlmostEqual(sum(out["weights"].values()), 1)
        self.assertNotIn("future", out["weights"])
        self.assertNotIn("unknown", out["weights"])
    def test_empty_abstains(self):
        self.assertEqual(route([], 100)["status"], "abstain")
    def test_invalid_evidence_abstains(self):
        for change in [dict(available_at=101), dict(expires_at=99),
                       dict(identified=False), dict(provenance=""),
                       dict(score=float("nan")), dict(penalty=-1)]:
            with self.subTest(change=change):
                self.assertEqual(route([replace(self.r, **change)],100)["status"],"abstain")
    def test_large_logits_do_not_overflow(self):
        out=route([replace(self.r,score=10000),replace(self.r,expert="b",score=9999)],100)
        self.assertAlmostEqual(sum(out["weights"].values()),1)
    def test_duplicates_rejected(self):
        with self.assertRaises(ValueError): route([self.r,self.r],100)

if __name__ == "__main__": unittest.main()
