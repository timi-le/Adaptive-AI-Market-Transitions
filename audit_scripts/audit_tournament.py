#!/usr/bin/env python3
"""Audit multi-model tournament exports for diversity, leakage, and timing issues."""

from __future__ import annotations

import argparse
import collections
import hashlib
import itertools
import json
import math
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from sklearn.metrics import cohen_kappa_score


DIRECTIONS = ("LONG", "SHORT", "NEUTRAL")
HIDDEN_CONTEXT_PATTERNS = {
    "alpha": re.compile(r"\balpha\b", re.IGNORECASE),
    "v3": re.compile(r"\bV3\b", re.IGNORECASE),
    "market_regime": re.compile(r"\b(?:market )?regime\b", re.IGNORECASE),
    "news": re.compile(r"\bnews\b", re.IGNORECASE),
    "signal_quality": re.compile(r"\bsignal quality\b", re.IGNORECASE),
}


def digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def parse_timestamp(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def pct(numerator: int, denominator: int) -> float | None:
    return 100.0 * numerator / denominator if denominator else None


def numeric(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "min": None, "mean": None, "max": None}
    return {"count": len(values), "min": min(values), "mean": sum(values) / len(values), "max": max(values)}


def load(path: Path, split: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    models: collections.Counter[str] = collections.Counter()
    directions: collections.Counter[str] = collections.Counter()
    consensus_directions: collections.Counter[str] = collections.Counter()
    by_model_direction: dict[str, collections.Counter[str]] = collections.defaultdict(collections.Counter)
    by_model_confidence: dict[str, list[float]] = collections.defaultdict(list)
    by_model_alignment: collections.Counter[str] = collections.Counter()
    hidden_mentions: collections.Counter[str] = collections.Counter()
    user_hashes: collections.Counter[str] = collections.Counter()
    decision_ids: collections.Counter[int] = collections.Counter()
    model_decision_pairs: collections.Counter[tuple[str, int]] = collections.Counter()
    assistant_json_failures = 0
    metadata_alignment_errors = 0
    prompt_metadata_consensus_errors = 0
    confidence_metadata_errors = 0
    response_before_prompt_timestamp = 0
    response_time_deltas_seconds: list[float] = []
    rows: list[dict[str, Any]] = []

    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, 1):
            if not raw_line.strip():
                continue
            record = json.loads(raw_line)
            messages = record.get("messages", [])
            metadata = record.get("metadata", {})
            user = str(messages[1].get("content", "")) if len(messages) > 1 else ""
            assistant = str(messages[2].get("content", "")) if len(messages) > 2 else ""
            try:
                output = json.loads(assistant)
            except json.JSONDecodeError:
                assistant_json_failures += 1
                continue

            model = str(metadata.get("model_name", "<missing>"))
            direction = str(output.get("direction", "<missing>"))
            confidence = float(output.get("confidence", math.nan))
            consensus = str(metadata.get("consensus_direction", "<missing>"))
            decision_id = int(metadata.get("v3_decision_id"))
            reasoning = str(output.get("reasoning", ""))
            models[model] += 1
            directions[direction] += 1
            consensus_directions[consensus] += 1
            by_model_direction[model][direction] += 1
            if math.isfinite(confidence):
                by_model_confidence[model].append(confidence)
            aligned = direction == consensus
            if aligned:
                by_model_alignment[model] += 1
            if bool(metadata.get("vote_aligns_with_consensus")) != aligned:
                metadata_alignment_errors += 1
            if math.isfinite(confidence) and not math.isclose(confidence, float(metadata.get("confidence", math.nan)), abs_tol=1e-9):
                confidence_metadata_errors += 1

            prompt_consensus = re.search(r"Multi-model consensus direction: ([A-Z]+)", user)
            if not prompt_consensus or prompt_consensus.group(1) != consensus:
                prompt_metadata_consensus_errors += 1
            for name, pattern in HIDDEN_CONTEXT_PATTERNS.items():
                if pattern.search(reasoning):
                    hidden_mentions[name] += 1

            user_hash = digest(user)
            user_hashes[user_hash] += 1
            decision_ids[decision_id] += 1
            model_decision_pairs[(model, decision_id)] += 1

            prompt_time_match = re.search(r"Timestamp: ([^\n]+)", user)
            responded_at = metadata.get("responded_at")
            time_delta = None
            if prompt_time_match and responded_at:
                time_delta = (parse_timestamp(str(responded_at)) - parse_timestamp(prompt_time_match.group(1))).total_seconds()
                response_time_deltas_seconds.append(time_delta)
                if time_delta < 0:
                    response_before_prompt_timestamp += 1

            rows.append({
                "split": split,
                "model": model,
                "direction": direction,
                "confidence": confidence,
                "consensus": consensus,
                "agreement_score": metadata.get("agreement_score"),
                "decision_id": decision_id,
                "user_hash": user_hash,
                "responded_at": metadata.get("responded_at"),
                "time_delta_seconds": time_delta,
            })

    duplicate_model_decision_rows = sum(count - 1 for count in model_decision_pairs.values() if count > 1)
    summary = {
        "path": str(path),
        "rows": len(rows),
        "models": dict(models),
        "directions": dict(directions),
        "consensus_directions": dict(consensus_directions),
        "unique_user_prompts": len(user_hashes),
        "user_prompts_repeated_for_multiple_targets": sum(1 for count in user_hashes.values() if count > 1),
        "unique_decision_ids": len(decision_ids),
        "responses_per_decision_id": dict(collections.Counter(decision_ids.values())),
        "duplicate_model_decision_rows": duplicate_model_decision_rows,
        "assistant_json_failures": assistant_json_failures,
        "metadata_alignment_errors": metadata_alignment_errors,
        "prompt_metadata_consensus_errors": prompt_metadata_consensus_errors,
        "confidence_metadata_errors": confidence_metadata_errors,
        "copy_consensus_baseline": {
            "correct": sum(by_model_alignment.values()),
            "accuracy_pct": pct(sum(by_model_alignment.values()), len(rows)),
        },
        "majority_direction_baseline": {
            "direction": directions.most_common(1)[0][0],
            "accuracy_pct": pct(directions.most_common(1)[0][1], len(rows)),
        },
        "by_model": {
            model: {
                "count": models[model],
                "directions": dict(by_model_direction[model]),
                "confidence": numeric(by_model_confidence[model]),
                "consensus_alignment_count": by_model_alignment[model],
                "consensus_alignment_pct": pct(by_model_alignment[model], models[model]),
            }
            for model in sorted(models)
        },
        "reasoning_mentions_context_absent_from_exported_prompt": {
            name: {"count": count, "pct": pct(count, len(rows))} for name, count in hidden_mentions.items()
        },
        "timing": {
            "response_before_prompt_timestamp_count": response_before_prompt_timestamp,
            "response_before_prompt_timestamp_pct": pct(response_before_prompt_timestamp, len(response_time_deltas_seconds)),
            "response_minus_prompt_seconds": numeric(response_time_deltas_seconds),
        },
    }
    return summary, rows


def pairwise_agreement(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    # Compare only outputs produced from exactly the same exported prompt. A
    # decision id can recur across replay runs with different timestamps or
    # aggregate context, so it is too coarse for policy-agreement estimates.
    by_prompt: dict[str, dict[str, str]] = collections.defaultdict(dict)
    for row in rows:
        by_prompt[row["user_hash"]][row["model"]] = row["direction"]
    models = sorted({row["model"] for row in rows})
    result = []
    for first, second in itertools.combinations(models, 2):
        pairs = [(votes[first], votes[second]) for votes in by_prompt.values() if first in votes and second in votes]
        left = [pair[0] for pair in pairs]
        right = [pair[1] for pair in pairs]
        agreement = sum(a == b for a, b in pairs)
        result.append({
            "model_a": first,
            "model_b": second,
            "shared_decisions": len(pairs),
            "raw_agreement_pct": pct(agreement, len(pairs)),
            "cohen_kappa": cohen_kappa_score(left, right, labels=list(DIRECTIONS)) if pairs else None,
        })
    return result


def group_agreement(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_prompt: dict[str, list[str]] = collections.defaultdict(list)
    for row in rows:
        by_prompt[row["user_hash"]].append(row["direction"])
    unanimity = 0
    distributions: collections.Counter[str] = collections.Counter()
    for votes in by_prompt.values():
        counts = sorted(collections.Counter(votes).values(), reverse=True)
        distributions["-".join(map(str, counts))] += 1
        if len(set(votes)) == 1:
            unanimity += 1
    return {
        "unique_prompts": len(by_prompt),
        "unanimous_decisions": unanimity,
        "unanimous_pct": pct(unanimity, len(by_prompt)),
        "vote_count_partition": dict(distributions),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--eval", type=Path, required=True)
    parser.add_argument("--quality-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    train_summary, train_rows = load(args.train, "train")
    eval_summary, eval_rows = load(args.eval, "eval")
    with args.quality_report.open("r", encoding="utf-8") as handle:
        declared = json.load(handle)
    all_rows = train_rows + eval_rows
    train_decisions = {row["decision_id"] for row in train_rows}
    eval_decisions = {row["decision_id"] for row in eval_rows}
    train_prompts = {row["user_hash"] for row in train_rows}
    eval_prompts = {row["user_hash"] for row in eval_rows}

    result = {
        "declared_quality_report": declared,
        "observed": {"train": train_summary, "eval": eval_summary},
        "cross_split": {
            "decision_id_overlap": len(train_decisions & eval_decisions),
            "exact_user_prompt_overlap": len(train_prompts & eval_prompts),
        },
        "combined": {
            "rows": len(all_rows),
            "pairwise_model_agreement": pairwise_agreement(all_rows),
            "group_agreement": group_agreement(all_rows),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")


if __name__ == "__main__":
    main()
