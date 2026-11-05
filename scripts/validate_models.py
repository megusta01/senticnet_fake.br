"""Nested CV selection of A/B/C and classifier, compared against length baseline."""
import argparse
import json
from pathlib import Path

import pandas as pd

from fake_news_detection.validation import nested_validate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=Path('data/analise_emocional_completa.csv'))
    parser.add_argument('--output', type=Path, default=Path('results/nested_validation.json'))
    parser.add_argument('--cv', type=int, default=5, help='Outer folds (default: 5)')
    parser.add_argument('--inner-cv', type=int, default=3, help='Inner folds (default: 3)')
    parser.add_argument('--fast', action='store_true', help='Reduce Random Forest hyperparameter grid')
    parser.add_argument('--n-jobs', type=int, default=1)
    parser.add_argument('--models', nargs='+', choices=['logistic_regression', 'random_forest'],
                        default=['logistic_regression', 'random_forest'])
    parser.add_argument('--no-group-duplicates', action='store_true',
                        help='Disable exact-text duplicate grouping (not recommended)')
    args = parser.parse_args()
    result = nested_validate(pd.read_csv(args.input), cv=args.cv, inner_cv=args.inner_cv,
        fast=args.fast, models=tuple(args.models), n_jobs=args.n_jobs,
        group_duplicates=not args.no_group_duplicates, checkpoint=args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f'\nResults written to {args.output}')
    print('model,metric,mean,std')
    for model, metrics in result['summary'].items():
        for metric, values in metrics.items():
            if isinstance(values, dict):
                print(f"{model},{metric},{values['mean']:.4f},{values['std']:.4f}")


if __name__ == '__main__':
    main()
