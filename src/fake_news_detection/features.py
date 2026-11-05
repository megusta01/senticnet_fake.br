"""Emotional features derived from a SenticNet dictionary.

Dictionary entries are ordered as described in the original TCC script:
[introspection, temper, attitude, sensitivity, primary_emotion, ...].
Tokenization follows the original whitespace-splitting implementation.
"""
from collections import Counter
from collections.abc import Mapping
from typing import Any

EMOTION_NAMES = {
    '#anger': 'contagem_raiva', '#grief': 'contagem_tristeza',
    '#fear': 'contagem_medo', '#joy': 'contagem_alegria',
    '#surprise': 'contagem_surpresa',
}


def extract_emotional_features(text: Any, lexicon: Mapping[str, list]) -> dict[str, int | float]:
    tokens = text.split() if isinstance(text, str) else []
    matched = [lexicon[token] for token in tokens if token in lexicon]
    emotions = Counter(entry[4] for entry in matched)
    n = len(matched)

    def average(index: int) -> float:
        return float(sum(entry[index] for entry in matched) / n) if n else 0.0

    return {
        'media_attitude': average(2),
        'media_sensitivity': average(3),
        'media_temper': average(1),
        'media_introspection': average(0),
        'total_palavras': len(tokens),
        'total_palavras_encontradas': n,
        **{column: emotions.get(emotion, 0) for emotion, column in EMOTION_NAMES.items()},
    }
