import numpy as np
import pandas as pd
from fake_news_detection.validation import baseline_matrix, nested_validate, text_groups


def make_data(n=36):
    rows = []
    for i in range(n):
        fake = i % 2 == 0
        rows.append(dict(label='fake' if fake else 'true',
                         preprocessed_news=f'noticia diferente {i} ' + ('raiva ' if fake else 'alegria ') * (i % 5 + 1),
                         media_sensitivity=float(fake), media_attitude=float(fake),
                         media_temper=float(fake), media_introspection=float(fake),
                         total_palavras_encontradas=i % 5 + 1,
                         contagem_raiva=int(fake) * (i % 5 + 1),
                         contagem_medo=int(fake), contagem_tristeza=int(fake),
                         contagem_alegria=int(not fake) * (i % 5 + 1),
                         contagem_surpresa=0))
    return pd.DataFrame(rows)


def test_audit_duplicates():
    df = make_data(8)
    df.loc[2, 'preprocessed_news'] = df.loc[0, 'preprocessed_news']
    groups, audit = text_groups(df)
    assert groups[0] == groups[2]
    assert audit['duplicate_rows'] == 1
    assert audit['conflicting_groups'] == 0


def test_length_baseline():
    X = baseline_matrix(make_data(10))
    assert list(X.columns) == ['total_palavras']
    assert np.all(X['total_palavras'] > 0)


def test_nested_small():
    output = nested_validate(make_data(), cv=2, inner_cv=2,
        fast=True, models=('logistic_regression',), n_jobs=1)
    assert len(output['folds']) == 2
    assert 'f1' in output['summary']['selected_model_outer']
    assert 'f1' in output['summary']['length_only_outer']
    assert sum(output['selection_counts'].values()) == 2
