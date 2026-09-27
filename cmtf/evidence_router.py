"""Demonstration of declared eligibility, not a causal-identification engine."""
from dataclasses import dataclass
from math import exp, isfinite
import json

@dataclass(frozen=True)
class Evidence:
    expert: str
    direction: int
    score: float
    response: float
    penalty: float
    available_at: int
    expires_at: int
    identified: bool
    provenance: str

def route(records, decision_at):
    """Reject invalid records; stable softmax over remaining declared logits."""
    if len({r.expert for r in records}) != len(records):
        raise ValueError("Duplicate expert identifier")
    eligible, rejected = [], {}
    for r in records:
        reason = None
        if not all(isfinite(x) for x in (r.score, r.response, r.penalty)):
            reason = "nonfinite"
        elif r.direction not in (-1, 0, 1) or r.penalty < 0:
            reason = "invalid_proposal"
        elif not r.provenance:
            reason = "missing_provenance"
        elif not r.identified:
            reason = "unidentified"
        elif not r.available_at <= decision_at <= r.expires_at:
            reason = "unavailable_or_expired"
        if reason:
            rejected[r.expert] = reason
        else:
            logit = r.score + r.direction * r.response - r.penalty
            if not isfinite(logit):
                rejected[r.expert] = "nonfinite_logit"
            else:
                eligible.append((r.expert, logit))
    if not eligible:
        return {"status": "abstain", "weights": {}, "rejected": rejected}
    largest = max(v for _, v in eligible)
    values = {k: exp(v-largest) for k, v in eligible}
    total = sum(values.values())
    return {"status": "routed", "weights": {k:v/total for k,v in values.items()},
            "rejected": rejected}

def example():
    return route([
        Evidence("long", 1, .2, -.8, .1, 90, 110, True, "synthetic:1"),
        Evidence("short", -1, .2, -.8, .1, 90, 110, True, "synthetic:1"),
        Evidence("future", 1, 100, 1, 0, 101, 110, True, "synthetic:2"),
        Evidence("unknown", 1, 100, 1, 0, 90, 110, False, "synthetic:3")], 100)

if __name__ == "__main__":
    print(json.dumps(example(), indent=2))
