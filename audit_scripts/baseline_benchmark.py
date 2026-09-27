#!/usr/bin/env python3
"""Fit transparent baselines to measure how much of the corpus is rule-recoverable.

These are dataset-diagnostic baselines, not trading-performance baselines.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor


def number(pattern: str, text: str) -> float:
    match = re.search(pattern, text)
    if not match:
        raise ValueError(f"Could not parse {pattern!r} from input")
    return float(match.group(1))


def word(pattern: str, text: str) -> str:
    match = re.search(pattern, text)
    if not match:
        raise ValueError(f"Could not parse {pattern!r} from input")
    return match.group(1)


def load(path: Path) -> dict[str, tuple[list[list[object]], list[object]]]:
    rows: dict[str, tuple[list[list[object]], list[object]]] = {
        "exit_timing": ([], []),
        "regime_classification": ([], []),
        "risk_sizing": ([], []),
    }
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            record = json.loads(raw_line)
            task = record["task"]
            if task not in rows:
                continue
            user = record["messages"][1]["content"]
            output = json.loads(record["messages"][2]["content"])
            features, targets = rows[task]
            if task == "exit_timing":
                features.append([
                    word(r"Position: ([A-Z]+)", user),
                    number(r"Unrealized P&L: ([+-]?[0-9.]+)%", user),
                    number(r"Hours Held: ([0-9.]+)", user),
                    word(r"Current Regime: ([A-Z_]+)", user),
                    number(r"Sell Quality: ([0-9.]+)", user),
                    number(r"Hurst Exponent: ([0-9.]+)", user),
                    number(r"Reversion Z: ([+-]?[0-9.]+)", user),
                ])
                targets.append(output["action"])
            elif task == "regime_classification":
                features.append([
                    number(r"log_return: ([+\-0-9.eE]+)", user),
                    number(r"volatility_ratio: ([+\-0-9.eE]+)", user),
                    number(r"momentum_gap: ([+\-0-9.eE]+)", user),
                    number(r"reversion_z: ([+\-0-9.eE]+)", user),
                    number(r"volume_ratio: ([+\-0-9.eE]+)", user),
                    number(r"hurst_exponent: ([+\-0-9.eE]+)", user),
                ])
                targets.append(output["regime"])
            elif task == "risk_sizing":
                features.append([
                    number(r"Account Equity: \$([0-9.]+)", user),
                    number(r"Base Risk: ([0-9.]+)%", user),
                    number(r"Win Rate \(30-trade window\): ([0-9.]+)%", user),
                    number(r"Avg Win/Loss Ratio: ([0-9.]+)", user),
                    number(r"Current Drawdown: ([0-9.]+)%", user),
                    word(r"Volatility Regime: ([A-Z_]+)", user),
                ])
                targets.append(output["final_risk_pct"])
    return rows


def classification_results(train_x, train_y, eval_x, eval_y, categorical, depths):
    results = []
    majority = max(set(train_y), key=train_y.count)
    majority_pred = [majority] * len(eval_y)
    results.append({
        "model": "majority",
        "accuracy": accuracy_score(eval_y, majority_pred),
        "macro_f1": f1_score(eval_y, majority_pred, average="macro", zero_division=0),
    })
    for depth in depths:
        transformer = ColumnTransformer(
            [("categorical", OneHotEncoder(handle_unknown="ignore"), categorical)],
            remainder="passthrough",
        )
        pipeline = Pipeline([
            ("features", transformer),
            ("tree", DecisionTreeClassifier(max_depth=depth, random_state=7)),
        ])
        pipeline.fit(train_x, train_y)
        prediction = pipeline.predict(eval_x)
        results.append({
            "model": f"decision_tree_depth_{depth if depth is not None else 'unbounded'}",
            "accuracy": accuracy_score(eval_y, prediction),
            "macro_f1": f1_score(eval_y, prediction, average="macro", zero_division=0),
        })
    return results


def regression_results(train_x, train_y, eval_x, eval_y, categorical, depths):
    results = []
    median = float(np.median(np.asarray(train_y, dtype=float)))
    median_pred = np.full(len(eval_y), median)
    results.append({
        "model": "train_median",
        "mae": mean_absolute_error(eval_y, median_pred),
        "r2": r2_score(eval_y, median_pred),
    })
    for depth in depths:
        transformer = ColumnTransformer(
            [("categorical", OneHotEncoder(handle_unknown="ignore"), categorical)],
            remainder="passthrough",
        )
        pipeline = Pipeline([
            ("features", transformer),
            ("tree", DecisionTreeRegressor(max_depth=depth, random_state=7)),
        ])
        pipeline.fit(train_x, train_y)
        prediction = pipeline.predict(eval_x)
        results.append({
            "model": f"decision_tree_depth_{depth if depth is not None else 'unbounded'}",
            "mae": mean_absolute_error(eval_y, prediction),
            "r2": r2_score(eval_y, prediction),
        })
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--eval", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    train = load(args.train)
    evaluation = load(args.eval)
    depths = [2, 3, 4, 6, 8, None]

    result = {
        "purpose": "Diagnostic test of label-rule recoverability; not evidence of live trading performance.",
        "exit_timing": classification_results(
            *train["exit_timing"], *evaluation["exit_timing"], categorical=[0, 3], depths=depths
        ),
        "regime_classification": classification_results(
            *train["regime_classification"], *evaluation["regime_classification"], categorical=[], depths=depths
        ),
        "risk_sizing": regression_results(
            *train["risk_sizing"], *evaluation["risk_sizing"], categorical=[5], depths=depths
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
        handle.write("\n")


if __name__ == "__main__":
    main()
