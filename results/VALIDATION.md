# Validação aninhada — V3

Este documento descreve o protocolo. **Não é relatório definitivo**; execute `scripts/validate_models.py` com 5 folds externos e os dois classificadores para gerar um JSON local com os resultados.

## Protocolo

- CV externa estratificada em 5 folds; CV interna estratificada em 3 folds.
- Comparação de normalização A/B/C e dos modelos LR/RF apenas dentro de cada fold externo de treinamento.
- Baseline: regressão logística usando somente o comprimento do texto pré-processado.
- Agrupamento de duplicatas *exatas* para impedir que o mesmo texto pré-processado apareça simultaneamente em treino e teste de um fold; não cobre duplicatas aproximadas.
- Saída: `results/nested_validation.json` (ignorado no Git).

## Execução de verificação já efetuada

Em 7.200 notícias, `--cv 2 --inner-cv 2 --models logistic_regression --fast` produziu F1 médio **0,9385** para a seleção interna de variantes e **0,9383** para comprimento isolado. Não equivale à configuração padrão completa nem a uma estimativa externa independente. O desempenho elevado do baseline de tamanho precisa ser investigado.

## Limitações

- Texto pré-processado, sem inspeção de duplicatas aproximadas ou agrupamento por notícia/evento.
- A métrica pode refletir diferença de comprimento entre classes, e não sinal emocional robusto.
- Resultados da exploração anterior já influenciaram escolhas metodológicas; generalização real exige avaliação em fontes independentes.
