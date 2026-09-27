#!/usr/bin/env python3
"""Descriptive agreement on one common exported-input panel; no return labels."""
import argparse
import collections
import json
from pathlib import Path

from audit_tournament import load, pairwise_agreement


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', type=Path, required=True)
    parser.add_argument('--eval', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    _, training = load(args.train, 'train')
    _, evaluation = load(args.eval, 'eval')
    rows = training + evaluation
    counts = collections.Counter((r['user_hash'], r['model']) for r in rows)
    duplicates = sum(n - 1 for n in counts.values() if n > 1)
    if duplicates:
        raise ValueError('Duplicate model/input pairs: resolve repeats before estimating agreement.')
    by_prompt = collections.defaultdict(dict)
    for row in rows:
        by_prompt[row['user_hash']][row['model']] = row['direction']
    models = sorted({row['model'] for row in rows})
    common = {p for p, votes in by_prompt.items() if set(votes) == set(models)}
    panel = [r for r in rows if r['user_hash'] in common]
    output = {
        'scope': 'Exact exported user-input matches; original generation inputs are unverified.',
        'duplicate_exported_prompt_model_rows': duplicates,
        'common_four_model_prompt_count': len(common),
        'panel_response_count_by_split': dict(collections.Counter(r['split'] for r in panel)),
        'panel_unique_decision_ids': len({r['decision_id'] for r in panel}),
        'common_panel_pairwise': pairwise_agreement(panel),
        'model_direction_counts': {m: dict(collections.Counter(r['direction'] for r in panel if r['model'] == m)) for m in models},
        'interpretation': 'Descriptive agreement on the selected export; neither accuracy nor causal effects of model architecture.'
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + '\n')


if __name__ == '__main__':
    main()
