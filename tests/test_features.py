from fake_news_detection.features import extract_emotional_features


def test_emotional_features():
    lexicon = {
        'feliz': [0.5, 0.1, 0.7, 0.9, '#joy'],
        'medo': [-0.2, 0.4, -0.7, 0.6, '#fear'],
    }
    features = extract_emotional_features('feliz desconhecido medo feliz', lexicon)
    assert features['total_palavras'] == 4
    assert features['total_palavras_encontradas'] == 3
    assert features['contagem_alegria'] == 2
    assert features['contagem_medo'] == 1
    assert features['media_attitude'] == (0.7 - 0.7 + 0.7) / 3


def test_empty_text():
    features = extract_emotional_features(None, {})
    assert features['total_palavras'] == 0
    assert features['media_attitude'] == 0
