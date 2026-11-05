"""Length-confounding audit and grouped nested-CV feature ablation.

Outer folds are shared by all feature sets. Feature-set-specific hyperparameters
are selected on inner folds only; no outer-test metrics are used for selection.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .preprocessing import COUNT_COLUMNS, MEAN_COLUMNS, build_feature_matrix
from .training import RANDOM_STATE
from .validation import baseline_matrix, scores, text_groups, _splitter

FEATURE_SETS = ('length_only', 'emotion_only', 'emotion_plus_length', 'means_only')
METRICS = ('accuracy', 'precision', 'recall', 'f1', 'roc_auc')


def ablation_matrices(df):
    """Emotion-only excludes raw counts and log-counts, but ratios can still encode length.

    This uses existing SenticNet average scores plus emotion counts / all tokens.
    The combined case adds the explicit total word count. Means-only is an
    additional sensitivity analysis without counts, ratios or explicit length.
    """
    b, labels = build_feature_matrix(df, variant='B')
    length = baseline_matrix(df)
    ratios = [f'{c}_prop' for c in COUNT_COLUMNS]
    emotion = b[MEAN_COLUMNS + ratios].copy()
    combined = emotion.copy()
    combined['total_palavras'] = length['total_palavras'].to_numpy()
    matrices = {
        'length_only': length,
        'emotion_only': emotion,
        'emotion_plus_length': combined,
        'means_only': b[MEAN_COLUMNS].copy(),
    }
    return matrices, labels.to_numpy()


def length_audit(df):
    lengths = baseline_matrix(df)['total_palavras'].to_numpy()
    labels = df['label'].to_numpy()
    if set(labels) != {'fake', 'true'}:
        raise ValueError('Expected both labels fake and true.')
    result = {}
    for label in ('fake', 'true'):
        x = lengths[labels == label]
        result[label] = {
            'n': int(len(x)), 'mean': float(np.mean(x)),
            'std': float(np.std(x, ddof=1)) if len(x) > 1 else 0.0,
            'median': float(np.median(x)), 'min': float(np.min(x)),
            'p25': float(np.percentile(x, 25)),
            'p75': float(np.percentile(x, 75)),
            'p90': float(np.percentile(x, 90)), 'max': float(np.max(x)),
        }
    return result


def _search(model, inner, fast, n_jobs):
    if model == 'logistic_regression':
        clf = LogisticRegression(max_iter=1500, class_weight='balanced', random_state=RANDOM_STATE)
        grid = {'clf__C': [0.1, 1, 10] if fast else [0.01, 0.1, 1, 10]}
    elif model == 'random_forest':
        clf = RandomForestClassifier(class_weight='balanced', random_state=RANDOM_STATE)
        grid = {'clf__n_estimators': [100] if fast else [100, 200],
                'clf__max_depth': [None] if fast else [None, 10, 20]}
    else:
        raise ValueError('Unsupported model: ' + model)
    estimator = Pipeline([('scale', StandardScaler()), ('clf', clf)])
    return GridSearchCV(estimator, grid, scoring='f1', cv=inner, n_jobs=n_jobs)


def run_ablation(df, *, cv=5, inner_cv=3, model='logistic_regression', fast=False,
                 n_jobs=1, group_duplicates=True, checkpoint=None):
    if cv < 2 or inner_cv < 2:
        raise ValueError('cv and inner_cv must be >= 2')
    matrices, y = ablation_matrices(df)
    groups, duplicate_audit = text_groups(df)
    if not group_duplicates:
        groups = None
    grouped = groups is not None
    outer = _splitter(cv, grouped, RANDOM_STATE)
    iterator = outer.split(matrices['length_only'], y, groups) if grouped else outer.split(matrices['length_only'], y)
    output = {
        'protocol': 'Shared grouped outer folds; independent inner hyperparameter tuning for each feature set.',
        'outer_folds': cv, 'inner_folds': inner_cv, 'model': model,
        'seed': RANDOM_STATE, 'fast_grid': bool(fast), 'n_rows': int(len(df)),
        'group_exact_duplicate_texts': grouped, 'duplicate_audit': duplicate_audit,
        'feature_sets': {key: list(frame.columns) for key, frame in matrices.items()},
        'length_audit': length_audit(df), 'folds': [],
        'limitations': [
            'The Fake.Br corpus can encode collection/source artifacts and text-length differences.',
            'Emotion-only ratios may still correlate with length; this is not a causal isolation of emotion.',
            'Only exact normalized-text duplicates are grouped, not similar or related articles.',
            'Performance on this corpus does not establish out-of-distribution generalization.',
            'Outer folds are evaluation folds, not an untouched external dataset.',
        ],
    }
    checkpoint = Path(checkpoint) if checkpoint is not None else None
    for fold, (train, test) in enumerate(iterator, 1):
        print(f'Outer fold {fold}/{cv}', flush=True)
        train_groups = groups[train] if grouped else None
        if grouped and set(groups[train]) & set(groups[test]):
            raise RuntimeError('Duplicate leakage across outer folds')
        inner = _splitter(inner_cv, grouped, RANDOM_STATE + fold)
        record = {'fold': fold, 'n_train': len(train), 'n_test': len(test), 'feature_sets': {}}
        for key, matrix in matrices.items():
            search = _search(model, inner, fast, n_jobs)
            fit_kw = {'groups': train_groups} if grouped else {}
            search.fit(matrix.iloc[train], y[train], **fit_kw)
            metrics = scores(search.best_estimator_, matrix.iloc[test], y[test])
            record['feature_sets'][key] = {
                'inner_f1': float(search.best_score_),
                'best_params': search.best_params_, 'outer': metrics,
            }
            print(f'  {key}: inner={search.best_score_:.4f}; outer F1={metrics["f1"]:.4f}', flush=True)
        output['folds'].append(record)
        if checkpoint:
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            checkpoint.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding='utf-8')
    summary = {}
    for key in FEATURE_SETS:
        entries = [f['feature_sets'][key]['outer'] for f in output['folds']]
        summary[key] = {metric: {
            'mean': float(np.mean([entry[metric] for entry in entries])),
            'std': float(np.std([entry[metric] for entry in entries], ddof=1))
        } for metric in METRICS}
        summary[key]['confusion_matrix_sum'] = np.sum(
            [entry['confusion_matrix'] for entry in entries], axis=0).tolist()
    output['summary'] = summary
    output['paired_f1_differences'] = {}
    for alternative in ('emotion_only', 'emotion_plus_length', 'means_only'):
        delta = np.array([
            f['feature_sets'][alternative]['outer']['f1'] -
            f['feature_sets']['length_only']['outer']['f1'] for f in output['folds']
        ])
        output['paired_f1_differences'][f'{alternative}_minus_length_only'] = {
            'mean': float(delta.mean()), 'std': float(delta.std(ddof=1)),
            'by_fold': delta.tolist(),
        }
    if checkpoint:
        checkpoint.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding='utf-8')
    return output


def report_markdown(output):
    lines = ['# Experimento de ablação — versão 4', '',
             'Os números são médias e desvios-padrão entre folds externos, **não** intervalos de confiança.', '',
             f'Protocolo: {output["outer_folds"]} folds externos, {output["inner_folds"]} internos; '
             f'modelo {output["model"]}; amostras {output["n_rows"]}.', '',
             '## Distribuição do comprimento', '',
             '| Classe | N | Média | Mediana | P25 | P75 | P90 |',
             '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for label, v in output['length_audit'].items():
        lines.append(f'| {label} | {v["n"]} | {v["mean"]:.1f} | {v["median"]:.1f} | {v["p25"]:.1f} | {v["p75"]:.1f} | {v["p90"]:.1f} |')
    lines += ['', '## Comparação de características', '',
              '| Conjunto | F1 (média ± DP) | Acurácia | ROC AUC |',
              '| --- | ---: | ---: | ---: |']
    if 'summary' in output:
        for key, m in output['summary'].items():
            lines.append(f'| {key} | {m["f1"]["mean"]:.4f} ± {m["f1"]["std"]:.4f} | '
                         f'{m["accuracy"]["mean"]:.4f} | {m["roc_auc"]["mean"]:.4f} |')
        lines += ['', '## Diferença pareada de F1 contra comprimento', '']
        for key, val in output['paired_f1_differences'].items():
            lines.append(f'- `{key}`: {val["mean"]:+.4f} (DP {val["std"]:.4f}).')
    lines += ['', '## Limitações', ''] + [f'- {lim}' for lim in output['limitations']]
    lines += ['', 'As análises são exploratórias e não estabelecem que emoções causaram o desempenho.', '']
    return '\n'.join(lines)
