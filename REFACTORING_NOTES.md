# Notas da primeira refatoração

- Fonte: `TCC.zip` entregue pelo autor.
- O script original `script/enriquecer_dados.py` carregava `pre-processed.csv` do diretório de execução e importava `senticnet_data.py` localmente.
- O notebook `lr_rf.ipynb` utilizava `total_palavras`, mas o extrator original não a gerava. Foi adicionada como **total de tokens do texto pré-processado**; isso altera o contrato dos dados e exige validação metodológica.
- O dicionário SenticNet (~5,9 MB), o corpus Fake.Br, cópias de CSVs e diretórios `.git` internos não estão incluídos nesta distribuição.
- Os notebooks selecionados são preservados com outputs removidos e caminhos adaptados para `data/`. Alguns notebooks de análise podem precisar de execução/ajustes manuais posteriores.
- O ensemble `VotingClassifier` existente no notebook original não foi incorporado ao script principal, que se concentra nos dois modelos destacados no TCC.
- Esta entrega **não comprova** reprodução dos valores do TCC. Testes unitários cobrem a extração e a preparação das features, sem avaliar o corpus.


## Comparação de normalizações (v2)
- A: sem proporções; B: denominador total de tokens de texto pré-processado; C: denominador total de tokens encontrados no SenticNet.
- Todos os cenários usam a mesma divisão 80/20 e os mesmos hiperparâmetros candidatos.
- A seleção definitiva da variante não deve ser feita usando o teste como conjunto de validação.
