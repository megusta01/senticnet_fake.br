# Comparação exploratória das normalizações (v2)

Fonte: CSV de características emocionais do TCC (7.200 notícias; corpus não redistribuído).

**Protocolo:** 80/20 estratificado (5.760/1.440), `random_state=42`, 5 folds estratificados para ajuste dos hiperparâmetros, mesmos índices em A/B/C. Para esta rodada, Random Forest usou apenas `n_estimators=100` e `max_depth=None` (busca reduzida), enquanto Regressão Logística testou quatro valores de C.

| Cenário | Modelo | F1 CV | Acurácia teste | F1 teste | ROC AUC teste |
|---|---|---:|---:|---:|---:|
| A | logistic_regression | 0.9248 | 0.9243 | 0.9253 | 0.9707 |
| A | random_forest | 0.9197 | 0.9236 | 0.9243 | 0.9683 |
| B | logistic_regression | 0.9390 | 0.9389 | 0.9392 | 0.9801 |
| B | random_forest | 0.9347 | 0.9424 | 0.9426 | 0.9804 |
| C | logistic_regression | 0.9311 | 0.9375 | 0.9384 | 0.9774 |
| C | random_forest | 0.9271 | 0.9354 | 0.9361 | 0.9767 |

A: sem proporções; B: emoções / tokens no texto pré-processado; C: emoções / tokens com correspondência no SenticNet. Em B, `total_palavras` foi calculado com `str.split()` sobre `preprocessed_news` do CSV original.

**Interpretação:** B teve a maior pontuação de teste, mas o conjunto de teste foi examinado nos três cenários e não deve ser usado para declarar uma escolha final sem novo protocolo de seleção. Os números anteriores do TCC podem ter sido produzidos com o denominador C, mas isso permanece uma hipótese.

**Limitações:** não houve auditoria de duplicações semânticas, viés de origem/comprimento ou validação em notícias externas ao corpus. Resultados obtidos a partir de atributos já extraídos do CSV original, não de uma nova execução SenticNet.
