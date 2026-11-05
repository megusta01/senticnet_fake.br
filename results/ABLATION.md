# Experimento de ablação — versão 4

Os números são médias e desvios-padrão entre folds externos, **não** intervalos de confiança.

Protocolo: 5 folds externos, 3 internos; modelo logistic_regression; amostras 7200.

## Distribuição do comprimento

| Classe | N | Média | Mediana | P25 | P75 | P90 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| fake | 3600 | 107.9 | 92.0 | 67.8 | 130.0 | 179.0 |
| true | 3600 | 624.9 | 523.0 | 364.0 | 806.0 | 1123.1 |

## Comparação de características

| Conjunto | F1 (média ± DP) | Acurácia | ROC AUC |
| --- | ---: | ---: | ---: |
| length_only | 0.9386 ± 0.0066 | 0.9374 | 0.9804 |
| emotion_only | 0.6075 ± 0.0132 | 0.6094 | 0.6399 |
| emotion_plus_length | 0.9401 ± 0.0058 | 0.9389 | 0.9807 |
| means_only | 0.5460 ± 0.0055 | 0.5490 | 0.5680 |

## Diferença pareada de F1 contra comprimento

- `emotion_only_minus_length_only`: -0.3311 (DP 0.0110).
- `emotion_plus_length_minus_length_only`: +0.0014 (DP 0.0042).
- `means_only_minus_length_only`: -0.3927 (DP 0.0095).

## Limitações

- The Fake.Br corpus can encode collection/source artifacts and text-length differences.
- Emotion-only ratios may still correlate with length; this is not a causal isolation of emotion.
- Only exact normalized-text duplicates are grouped, not similar or related articles.
- Performance on this corpus does not establish out-of-distribution generalization.
- Outer folds are evaluation folds, not an untouched external dataset.

As análises são exploratórias e não estabelecem que emoções causaram o desempenho.

