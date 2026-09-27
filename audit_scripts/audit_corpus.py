#!/usr/bin/env python3
"""Audit ATRX JSONL instruction corpora and emit reproducible summary artifacts.

The script uses only Python's standard library so it can run in CI without
project-specific dependencies.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any


SYSTEM_PROMPT = (
    "You are ATRX-1B, a quantitative trading decision engine. You analyze "
    "market microstructure, regime states, and statistical signals to produce "
    "structured JSON decisions. Be precise, quantitative, and decisive."
)


def digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = (len(ordered) - 1) * q
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - index) + ordered[upper] * (index - lower)


def numeric_summary(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "min": None, "p25": None, "median": None, "p75": None, "max": None, "mean": None}
    return {
        "count": len(values),
        "min": min(values),
        "p25": percentile(values, 0.25),
        "median": percentile(values, 0.5),
        "p75": percentile(values, 0.75),
        "max": max(values),
        "mean": sum(values) / len(values),
    }


def extract_number(pattern: str, text: str) -> float | None:
    match = re.search(pattern, text)
    return float(match.group(1)) if match else None


def parse_record(line: str, source: Path, line_number: int) -> dict[str, Any]:
    try:
        record = json.loads(line)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {source}:{line_number}: {exc}") from exc
    if not isinstance(record, dict):
        raise ValueError(f"Record is not an object in {source}:{line_number}")
    return record


def audit_split(path: Path) -> tuple[dict[str, Any], set[str], set[str]]:
    task_counts: collections.Counter[str] = collections.Counter()
    action_counts: collections.Counter[str] = collections.Counter()
    regime_counts: collections.Counter[str] = collections.Counter()
    volatility_counts: collections.Counter[str] = collections.Counter()
    role_patterns: collections.Counter[tuple[str, ...]] = collections.Counter()
    assistant_keys: dict[str, collections.Counter[str]] = collections.defaultdict(collections.Counter)
    system_prompts: collections.Counter[str] = collections.Counter()
    record_hashes: set[str] = set()
    user_hashes: set[str] = set()
    duplicate_records = 0
    duplicate_inputs = 0
    assistant_json_failures = 0
    missing_or_extra_message_count = 0
    regime_probability_sum_errors = 0
    regime_argmax_errors = 0
    risk_amount_errors = 0
    kelly_full_errors = 0
    reasoning_fraction_mismatches = 0
    confidences: dict[str, list[float]] = collections.defaultdict(list)
    risk_final_values: list[float] = []
    kelly_full_values: list[float] = []
    kelly_fraction_values: list[float] = []
    risk_output_rows = 0
    negative_kelly_rows = 0
    final_risk_floor_rows = 0
    kelly_fraction_floor_rows = 0
    line_count = 0

    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, 1):
            if not raw_line.strip():
                continue
            line_count += 1
            record = parse_record(raw_line, path, line_number)
            normalized = json.dumps(record, sort_keys=True, separators=(",", ":"))
            record_hash = digest(normalized)
            if record_hash in record_hashes:
                duplicate_records += 1
            record_hashes.add(record_hash)

            task = str(record.get("task", "<missing>"))
            task_counts[task] += 1
            messages = record.get("messages", [])
            if not isinstance(messages, list) or len(messages) != 3:
                missing_or_extra_message_count += 1
                continue
            roles = tuple(str(message.get("role", "<missing>")) for message in messages)
            role_patterns[roles] += 1
            system_text = str(messages[0].get("content", ""))
            user_text = str(messages[1].get("content", ""))
            assistant_text = str(messages[2].get("content", ""))
            system_prompts[system_text] += 1
            user_hash = digest(user_text)
            if user_hash in user_hashes:
                duplicate_inputs += 1
            user_hashes.add(user_hash)

            try:
                output = json.loads(assistant_text)
            except (json.JSONDecodeError, TypeError):
                assistant_json_failures += 1
                continue
            if not isinstance(output, dict):
                assistant_json_failures += 1
                continue
            assistant_keys[task].update(output.keys())
            confidence = output.get("confidence")
            if isinstance(confidence, (int, float)):
                confidences[task].append(float(confidence))

            if task == "exit_timing":
                action_counts[str(output.get("action", "<missing>"))] += 1

            elif task == "regime_classification":
                regime = str(output.get("regime", "<missing>"))
                regime_counts[regime] += 1
                probabilities = output.get("regime_probabilities")
                if isinstance(probabilities, dict) and probabilities:
                    numeric = {str(k): float(v) for k, v in probabilities.items() if isinstance(v, (int, float))}
                    if not math.isclose(sum(numeric.values()), 1.0, abs_tol=0.002):
                        regime_probability_sum_errors += 1
                    if numeric and max(numeric, key=numeric.get) != regime:
                        regime_argmax_errors += 1

            elif task == "risk_sizing":
                risk_output_rows += 1
                volatility_match = re.search(r"Volatility Regime: ([A-Z_]+)", user_text)
                if volatility_match:
                    volatility_counts[volatility_match.group(1)] += 1
                equity = extract_number(r"Account Equity: \$([0-9.]+)", user_text)
                win_rate_pct = extract_number(r"Win Rate \(30-trade window\): ([0-9.]+)%", user_text)
                win_loss = extract_number(r"Avg Win/Loss Ratio: ([0-9.]+)", user_text)
                final_risk = output.get("final_risk_pct")
                risk_amount = output.get("risk_amount_usd")
                kelly_full = output.get("kelly_full")
                kelly_fraction = output.get("kelly_fraction")
                if isinstance(final_risk, (int, float)):
                    risk_final_values.append(float(final_risk))
                    if math.isclose(float(final_risk), 0.05, abs_tol=1e-12):
                        final_risk_floor_rows += 1
                if isinstance(kelly_full, (int, float)):
                    kelly_full_values.append(float(kelly_full))
                    if float(kelly_full) < 0:
                        negative_kelly_rows += 1
                if isinstance(kelly_fraction, (int, float)):
                    kelly_fraction_values.append(float(kelly_fraction))
                    if math.isclose(float(kelly_fraction), 0.1, abs_tol=1e-12):
                        kelly_fraction_floor_rows += 1
                if all(isinstance(v, (int, float)) for v in (equity, final_risk, risk_amount)):
                    expected_amount = float(equity) * float(final_risk) / 100.0
                    if not math.isclose(expected_amount, float(risk_amount), abs_tol=0.011):
                        risk_amount_errors += 1
                if all(isinstance(v, (int, float)) for v in (win_rate_pct, win_loss, kelly_full)):
                    p = float(win_rate_pct) / 100.0
                    expected_kelly = p - (1.0 - p) / float(win_loss)
                    if not math.isclose(expected_kelly, float(kelly_full), abs_tol=0.00015):
                        kelly_full_errors += 1
                reasoning = str(output.get("reasoning", ""))
                fraction_match = re.search(r"fractional \(0\.25x\) = (-?\d+(?:\.\d+)?)", reasoning)
                if fraction_match and isinstance(kelly_fraction, (int, float)):
                    stated_fraction = float(fraction_match.group(1))
                    if not math.isclose(stated_fraction, float(kelly_fraction), abs_tol=0.0011):
                        reasoning_fraction_mismatches += 1

    summary = {
        "path": str(path),
        "rows": line_count,
        "tasks": dict(task_counts),
        "role_patterns": {"|".join(k): v for k, v in role_patterns.items()},
        "unique_system_prompts": len(system_prompts),
        "canonical_system_prompt_rows": system_prompts[SYSTEM_PROMPT],
        "assistant_json_failures": assistant_json_failures,
        "message_count_anomalies": missing_or_extra_message_count,
        "duplicate_full_records_within_split": duplicate_records,
        "duplicate_user_inputs_within_split": duplicate_inputs,
        "assistant_keys": {task: dict(keys) for task, keys in assistant_keys.items()},
        "exit_actions": dict(action_counts),
        "regime_labels": dict(regime_counts),
        "risk_volatility_regimes": dict(volatility_counts),
        "confidence": {task: numeric_summary(values) for task, values in confidences.items()},
        "regime_probability_sum_errors": regime_probability_sum_errors,
        "regime_argmax_errors": regime_argmax_errors,
        "risk_sizing": {
            "rows": risk_output_rows,
            "final_risk_pct": numeric_summary(risk_final_values),
            "kelly_full": numeric_summary(kelly_full_values),
            "kelly_fraction": numeric_summary(kelly_fraction_values),
            "negative_full_kelly_rows": negative_kelly_rows,
            "final_risk_at_0_05_pct_floor_rows": final_risk_floor_rows,
            "kelly_fraction_at_0_1_floor_rows": kelly_fraction_floor_rows,
            "risk_amount_arithmetic_errors": risk_amount_errors,
            "kelly_full_formula_errors": kelly_full_errors,
            "reasoning_vs_kelly_fraction_mismatches": reasoning_fraction_mismatches,
        },
    }
    return summary, record_hashes, user_hashes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--eval", type=Path, required=True)
    parser.add_argument("--dataset-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    train_summary, train_records, train_inputs = audit_split(args.train)
    eval_summary, eval_records, eval_inputs = audit_split(args.eval)
    with args.dataset_report.open("r", encoding="utf-8") as handle:
        declared_report = json.load(handle)

    result = {
        "declared_dataset_report": declared_report,
        "observed": {"train": train_summary, "eval": eval_summary},
        "cross_split": {
            "exact_full_record_overlap": len(train_records & eval_records),
            "exact_user_input_overlap": len(train_inputs & eval_inputs),
        },
        "checks": {
            "declared_total_matches": declared_report.get("total_examples") == train_summary["rows"] + eval_summary["rows"],
            "declared_train_matches": declared_report.get("train_size") == train_summary["rows"],
            "declared_eval_matches": declared_report.get("eval_size") == eval_summary["rows"],
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")


if __name__ == "__main__":
    main()
