import pandas as pd
import pytest
from fake_news_detection.preprocessing import build_feature_matrix


def test_proportions_zero_safe():
    row = {'label': 'fake', 'total_palavras': 0}
    row.update({k: 0 for k in ['media_sensitivity','media_attitude','media_temper','media_introspection',
                'contagem_raiva','contagem_medo','contagem_tristeza','contagem_alegria','contagem_surpresa']})
    features, labels = build_feature_matrix(pd.DataFrame([row]))
    assert features['contagem_raiva_prop'].iloc[0] == 0
    assert labels.iloc[0] == 1


def test_missing_total_words_rejected():
    with pytest.raises(ValueError, match='total_palavras'):
        build_feature_matrix(pd.DataFrame({'label': ['fake']}))
