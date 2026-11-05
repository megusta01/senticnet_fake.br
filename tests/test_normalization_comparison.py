import pandas as pd
import pytest
from fake_news_detection.preprocessing import build_feature_matrix


def sample():
    return pd.DataFrame([
        {'label': 'fake', 'preprocessed_news': 'a b c d',
         'total_palavras_encontradas': 2, 'contagem_raiva': 1},
        {'label': 'true', 'preprocessed_news': '',
         'total_palavras_encontradas': 0, 'contagem_raiva': 0},
    ]).assign(**{
        **{key: 0.0 for key in ['media_sensitivity', 'media_attitude', 'media_temper', 'media_introspection']},
        **{key: 0 for key in ['contagem_medo', 'contagem_tristeza', 'contagem_alegria', 'contagem_surpresa']},
    })


def test_variant_a_has_no_proportion():
    X, y = build_feature_matrix(sample(), variant='A')
    assert X.shape[1] == 14
    assert 'contagem_raiva_prop' not in X
    assert list(y) == [1, 0]


def test_variant_b_divides_by_all_tokens_and_handles_zero():
    X, _ = build_feature_matrix(sample(), variant='B')
    assert list(X['contagem_raiva_prop']) == [0.25, 0.0]


def test_variant_c_divides_by_matched_tokens_and_handles_zero():
    X, _ = build_feature_matrix(sample(), variant='C')
    assert list(X['contagem_raiva_prop']) == [0.5, 0.0]


def test_variant_b_accepts_explicit_total_words():
    df = sample().drop(columns='preprocessed_news').assign(total_palavras=[8, 0])
    X, _ = build_feature_matrix(df, variant='B')
    assert X['contagem_raiva_prop'].iloc[0] == 0.125


def test_missing_c_denominator_fails():
    with pytest.raises(ValueError, match='total_palavras_encontradas'):
        build_feature_matrix(sample().drop(columns='total_palavras_encontradas'), variant='C')


def test_invalid_variant_fails():
    with pytest.raises(ValueError, match='variant'):
        build_feature_matrix(sample(), variant='D')
