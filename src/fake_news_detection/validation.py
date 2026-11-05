"""Nested cross-validation with optional exact-text grouping and a length baseline.

The outer CV estimates the *selection procedure*: both normalisation and
hyperparameters are chosen exclusively within each outer training fold.
"""
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .preprocessing import build_feature_matrix
from .training import make_searches, RANDOM_STATE


def text_groups(df: pd.DataFrame):
    """Exact duplicate groups after trimming and normalizing whitespace."""
    if 'preprocessed_news' not in df:
        return None, {'available': False, 'duplicate_rows': None, 'conflicting_groups': None}
    normalized = df['preprocessed_news'].fillna('').astype(str).str.split().str.join(' ')
    # Do not merge all missing-text documents into one artificial group.
    keys = [sha256((s if s else f'__missing_row_{i}').encode('utf-8')).hexdigest()
            for i, s in enumerate(normalized)]
    counts = Counter(keys)
    grouped_labels = pd.DataFrame({'group': keys, 'label': df['label'].to_numpy()})
    conflicting = int((grouped_labels.groupby('group')['label'].nunique() > 1).sum())
    return np.asarray(keys), {
        'available': True, 'duplicate_rows': int(sum(v-1 for v in counts.values() if v > 1)),
        'duplicate_groups': int(sum(v > 1 for v in counts.values())),
        'conflicting_groups': conflicting,
        'warning': 'Only exact whitespace-normalized matches are checked; near-duplicates and related articles are not detected.'
    }


def scores(estimator, X, y):
    pred = estimator.predict(X)
    prob = estimator.predict_proba(X)[:, 1]
    return {
        'accuracy': float(accuracy_score(y, pred)),
        'precision': float(precision_score(y, pred, zero_division=0)),
        'recall': float(recall_score(y, pred, zero_division=0)),
        'f1': float(f1_score(y, pred, zero_division=0)),
        'roc_auc': float(roc_auc_score(y, prob)),
        'confusion_matrix': confusion_matrix(y, pred, labels=[0, 1]).tolist(),
    }


def baseline_matrix(df):
    if 'total_palavras' in df:
        lengths = pd.to_numeric(df['total_palavras'], errors='raise')
    elif 'preprocessed_news' in df:
        lengths = df['preprocessed_news'].fillna('').astype(str).str.split().str.len()
    else:
        raise ValueError('Length baseline requires total_palavras or preprocessed_news.')
    if lengths.isna().any() or (lengths < 0).any():
        raise ValueError('Invalid word counts for baseline.')
    # 1 dimension: explicitly tests whether length alone predicts the label.
    return pd.DataFrame({'total_palavras': lengths.to_numpy(dtype=float)})


def _splitter(n, grouped, seed):
    cls = StratifiedGroupKFold if grouped else StratifiedKFold
    return cls(n_splits=n, shuffle=True, random_state=seed)


def _search(name, inner_cv, fast, n_jobs):
    if name == 'length_baseline':
        model = Pipeline([
            ('scale', StandardScaler()),
            ('clf', LogisticRegression(max_iter=1000, class_weight='balanced', random_state=RANDOM_STATE)),
        ])
        return GridSearchCV(model, {'clf__C': [0.1, 1.0] if fast else [0.01, 0.1, 1, 10]},
                            scoring='f1', cv=inner_cv, n_jobs=n_jobs)
    search = make_searches(n_jobs=n_jobs)[name]
    search.cv = inner_cv
    if fast and name == 'random_forest':
        search.param_grid = {'clf__n_estimators': [100], 'clf__max_depth': [None]}
    return search


def _fit(search, X, y, groups):
    if groups is None:
        return search.fit(X, y)
    return search.fit(X, y, groups=groups)


def nested_validate(df: pd.DataFrame, *, cv: int = 5, inner_cv: int = 3,
                    models=('logistic_regression', 'random_forest'), fast=False,
                    n_jobs=1, group_duplicates=True, checkpoint: Path | None = None):
    if cv < 2 or inner_cv < 2:
        raise ValueError('Both cv and inner_cv must be >= 2.')
    if not models or set(models) - {'logistic_regression', 'random_forest'}:
        raise ValueError('Models must contain logistic_regression and/or random_forest.')
    matrices = {v: build_feature_matrix(df, variant=v) for v in ('A', 'B', 'C')}
    y = matrices['A'][1].to_numpy()
    baseline = baseline_matrix(df)
    groups, audit = text_groups(df)
    if not group_duplicates:
        groups = None
    grouped = groups is not None
    outer = _splitter(cv, grouped, RANDOM_STATE)
    split_iter = outer.split(matrices['A'][0], y, groups) if grouped else outer.split(matrices['A'][0], y)
    output = {
        'protocol': 'Nested stratified CV; each outer fold picks best (variant, model, hyperparameters) using inner training-only CV. Separate length-only baseline.',
        'outer_folds': cv, 'inner_folds': inner_cv, 'fast_grid': fast, 'seed': RANDOM_STATE,
        'n_rows': int(len(df)), 'group_exact_duplicate_texts': grouped,
        'duplicate_audit': audit, 'models': list(models), 'folds': [],
        'warning': 'Historical exploratory 80/20 test results were already inspected; this estimates the model-selection procedure on this corpus, not independent external generalization.',
    }
    for i, (train, test) in enumerate(split_iter, 1):
        print(f'Outer fold {i}/{cv}', flush=True)
        inner = _splitter(inner_cv, grouped, RANDOM_STATE + i)
        train_groups = groups[train] if grouped else None
        if grouped and set(groups[train]) & set(groups[test]):
            raise RuntimeError('Exact duplicate leakage detected in outer fold.')
        best = None
        for variant in ('A', 'B', 'C'):
            X = matrices[variant][0]
            for name in models:
                search = _search(name, inner, fast, n_jobs)
                _fit(search, X.iloc[train], y[train], train_groups)
                if best is None or search.best_score_ > best['inner_f1']:
                    best = {'variant': variant, 'model': name,
                            'inner_f1': float(search.best_score_),
                            'params': search.best_params_, 'estimator': search.best_estimator_}
        Xwinner = matrices[best['variant']][0]
        outer_scores = scores(best['estimator'], Xwinner.iloc[test], y[test])
        base_search = _search('length_baseline', inner, fast, n_jobs)
        _fit(base_search, baseline.iloc[train], y[train], train_groups)
        base_scores = scores(base_search.best_estimator_, baseline.iloc[test], y[test])
        record = {
            'fold': i, 'n_train': int(len(train)), 'n_test': int(len(test)),
            'selected_variant': best['variant'], 'selected_model': best['model'],
            'inner_f1': best['inner_f1'], 'best_params': best['params'],
            'selected_model_outer': outer_scores,
            'length_only_outer': base_scores,
            'length_baseline_params': base_search.best_params_,
        }
        output['folds'].append(record)
        if checkpoint:
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            checkpoint.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding='utf-8')
        print(f"  chose {best['variant']} / {best['model']}; outer F1={outer_scores['f1']:.4f}; length F1={base_scores['f1']:.4f}", flush=True)
    output['summary'] = {}
    for key in ('selected_model_outer', 'length_only_outer'):
        output['summary'][key] = {}
        for metric in ('accuracy', 'precision', 'recall', 'f1', 'roc_auc'):
            values = [f[key][metric] for f in output['folds']]
            output['summary'][key][metric] = {'mean': float(np.mean(values)), 'std': float(np.std(values, ddof=1))}
        output['summary'][key]['confusion_matrix_sum'] = np.sum(
            [f[key]['confusion_matrix'] for f in output['folds']], axis=0).tolist()
    output['selection_counts'] = {
        f'{v}/{m}': sum(f['selected_variant'] == v and f['selected_model'] == m for f in output['folds'])
        for v in ('A', 'B', 'C') for m in models
    }
    if checkpoint:
        checkpoint.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding='utf-8')
    return output
