"""Train models and save metrics; source notebook: lr_rf.ipynb."""
import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from fake_news_detection.preprocessing import build_feature_matrix
from fake_news_detection.training import make_searches, RANDOM_STATE
from fake_news_detection.evaluation import evaluate


def main():
    parser = argparse.ArgumentParser(description='Train and evaluate emotional fake-news classifiers.')
    parser.add_argument('--input', type=Path, default=Path('data/analise_emocional_completa.csv'))
    parser.add_argument('--output', type=Path, default=Path('results/metrics.json'))
    parser.add_argument('--n-jobs', type=int, default=-1)
    args = parser.parse_args()
    df = pd.read_csv(args.input)
    X, y = build_feature_matrix(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)
    metrics = {}
    for name, search in make_searches(n_jobs=args.n_jobs).items():
        print(f'Training {name}...')
        search.fit(X_train, y_train)
        metrics[name] = {
            'best_params': search.best_params_,
            'cv_f1': float(search.best_score_),
            'test': evaluate(search.best_estimator_, X_test, y_test),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f'Saved results to {args.output}')


if __name__ == '__main__':
    main()
