"""Compare A/B/C on identical held-out rows and CV folds.

Input is the original enriched CSV, which is NOT redistributed with this repo.
This comparison does not require the proprietary SenticNet lexicon.
"""
import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from fake_news_detection.evaluation import evaluate
from fake_news_detection.preprocessing import build_feature_matrix
from fake_news_detection.training import RANDOM_STATE, make_searches


def compare(df: pd.DataFrame, *, n_jobs: int = 1, models: tuple[str, ...] = ('logistic_regression', 'random_forest'), fast: bool = False, checkpoint: Path | None = None) -> dict:
    split_indices = pd.Series(range(len(df)))
    labels = df['label'].map({'fake': 1, 'true': 0})
    if labels.isna().any():
        raise ValueError('Unknown label.')
    train_idx, test_idx = train_test_split(
        split_indices, test_size=0.2, random_state=RANDOM_STATE, stratify=labels)
    summary = {
        'method': '80/20 stratified split; 5-fold stratified CV, same indices and random seeds for all variants',
        'random_state': RANDOM_STATE,
        'search_mode': 'reduced (RF 100 trees, unrestricted depth; LR 4 C values)' if fast else 'full TCC-aligned grid',
        'n_rows': len(df),
        'n_train': len(train_idx),
        'n_test': len(test_idx),
        'variants': {},
    }
    for variant in ('A', 'B', 'C'):
        X, y = build_feature_matrix(df, variant=variant)
        summary['variants'][variant] = {}
        searches = make_searches(n_jobs=n_jobs)
        if fast:
            searches['random_forest'].param_grid = {'clf__n_estimators': [100], 'clf__max_depth': [None]}
        for name in models:
            if name not in searches:
                raise ValueError(f'Unknown model: {name}')
            print(f'Variant {variant} / {name}', flush=True)
            search = searches[name]
            search.fit(X.iloc[train_idx], y.iloc[train_idx])
            summary['variants'][variant][name] = {
                'n_features': len(X.columns),
                'features': X.columns.tolist(),
                'cv_f1': float(search.best_score_),
                'best_params': search.best_params_,
                'test': evaluate(search.best_estimator_, X.iloc[test_idx], y.iloc[test_idx]),
            }
            if checkpoint:
                checkpoint.parent.mkdir(parents=True, exist_ok=True)
                checkpoint.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=Path('data/analise_emocional_completa.csv'))
    parser.add_argument('--output', type=Path, default=Path('results/normalization_comparison.json'))
    parser.add_argument('--n-jobs', type=int, default=1)
    parser.add_argument('--fast', action='store_true', help='Reduced RF grid for a faster preliminary comparison')
    parser.add_argument('--models', nargs='+', choices=['logistic_regression', 'random_forest'], default=['logistic_regression', 'random_forest'])
    args = parser.parse_args()
    comparison = compare(pd.read_csv(args.input), n_jobs=args.n_jobs, models=tuple(args.models), fast=args.fast, checkpoint=args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(comparison, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'Results written to {args.output}', flush=True)
    print('variant,model,cv_f1,test_accuracy,test_f1,test_roc_auc')
    for variant, record in comparison['variants'].items():
        for model, values in record.items():
            t = values['test']
            print(f"{variant},{model},{values['cv_f1']:.4f},{t['accuracy']:.4f},{t['f1']:.4f},{t['roc_auc']:.4f}")


if __name__ == '__main__':
    main()
