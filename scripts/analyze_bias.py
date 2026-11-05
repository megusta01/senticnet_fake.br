"""Paired nested-CV ablation of length and emotion features, plus a length audit."""
import argparse
import json
from pathlib import Path

import pandas as pd
from fake_news_detection.ablation import report_markdown, run_ablation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=Path('data/analise_emocional_completa.csv'))
    parser.add_argument('--output', type=Path, default=Path('results/ablation_validation.json'))
    parser.add_argument('--report', type=Path, default=Path('results/ABLATION_REPORT.md'))
    parser.add_argument('--cv', type=int, default=5)
    parser.add_argument('--inner-cv', type=int, default=3)
    parser.add_argument('--model', choices=['logistic_regression', 'random_forest'], default='logistic_regression')
    parser.add_argument('--fast', action='store_true')
    parser.add_argument('--n-jobs', type=int, default=2)
    parser.add_argument('--no-group-duplicates', action='store_true', help='Not recommended')
    args = parser.parse_args()
    data = pd.read_csv(args.input)
    result = run_ablation(data, cv=args.cv, inner_cv=args.inner_cv, model=args.model,
                          fast=args.fast, n_jobs=args.n_jobs,
                          group_duplicates=not args.no_group_duplicates, checkpoint=args.output)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(report_markdown(result), encoding='utf-8')
    print(f'\nResults written to {args.output} and {args.report}')
    print('feature_set,f1_mean,f1_std,accuracy,roc_auc')
    for key, stats in result['summary'].items():
        print(f'{key},{stats["f1"]["mean"]:.4f},{stats["f1"]["std"]:.4f},'
              f'{stats["accuracy"]["mean"]:.4f},{stats["roc_auc"]["mean"]:.4f}')


if __name__ == '__main__':
    main()
