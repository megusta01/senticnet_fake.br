"""Helpers for assembling the modeling feature matrix."""
import numpy as np
import pandas as pd

COUNT_COLUMNS = ['contagem_raiva', 'contagem_medo', 'contagem_tristeza',
                 'contagem_alegria', 'contagem_surpresa']
MEAN_COLUMNS = ['media_sensitivity', 'media_attitude',
                'media_temper', 'media_introspection']
FEATURE_COLUMNS = (MEAN_COLUMNS + COUNT_COLUMNS
                   + [f'{col}_log' for col in COUNT_COLUMNS]
                   + [f'{col}_prop' for col in COUNT_COLUMNS])


def build_feature_matrix(df: pd.DataFrame, *, variant: str = 'B') -> tuple[pd.DataFrame, pd.Series]:
    """Build comparable features for one of three normalization experiments.

    A: means, raw counts and log1p(counts), without proportions.
    B: A plus count / number of whitespace tokens in preprocessed_news.
    C: A plus count / number of SenticNet-matched tokens.

    Zero denominators yield zero proportions. The same non-proportion
    features are retained in all variants. Rows are never discarded.
    """
    if variant not in ('A', 'B', 'C'):
        raise ValueError('variant must be A, B or C')
    required = COUNT_COLUMNS + MEAN_COLUMNS + ['label']
    if variant == 'B' and 'total_palavras' not in df.columns and 'preprocessed_news' not in df.columns:
        raise ValueError('Variant B requires total_palavras or preprocessed_news.')
    if variant == 'C':
        required += ['total_palavras_encontradas']
    missing = set(required) - set(df.columns)
    if missing:
        raise ValueError(f'Missing required columns: {sorted(missing)}.')
    frame = df.copy()
    if variant == 'B' and 'total_palavras' not in frame:
        frame['total_palavras'] = frame['preprocessed_news'].fillna('').astype(str).str.split().str.len()
    numerics = COUNT_COLUMNS + MEAN_COLUMNS
    if variant == 'B':
        numerics += ['total_palavras']
    if variant == 'C':
        numerics += ['total_palavras_encontradas']
    if frame[numerics].isna().any().any():
        raise ValueError('Input contains null feature values.')
    if not np.isfinite(frame[numerics].to_numpy(dtype=float)).all():
        raise ValueError('Features must be finite.')
    if (frame[COUNT_COLUMNS] < 0).any().any():
        raise ValueError('Emotion counts cannot be negative.')
    for col in COUNT_COLUMNS:
        frame[f'{col}_log'] = np.log1p(frame[col])
    selected = MEAN_COLUMNS + COUNT_COLUMNS + [f'{col}_log' for col in COUNT_COLUMNS]
    if variant != 'A':
        denominator_col = 'total_palavras' if variant == 'B' else 'total_palavras_encontradas'
        denominator = frame[denominator_col]
        if (denominator < 0).any():
            raise ValueError('Word counts cannot be negative.')
        for col in COUNT_COLUMNS:
            frame[f'{col}_prop'] = (frame[col] / denominator.replace(0, np.nan)).fillna(0)
        selected += [f'{col}_prop' for col in COUNT_COLUMNS]
    labels = frame['label'].map({'fake': 1, 'true': 0})
    if labels.isna().any():
        raise ValueError('Unknown label. Expected fake or true.')
    return frame[selected], labels.astype(int)
