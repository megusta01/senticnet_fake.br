import numpy as np
import pandas as pd
import pytest
from fake_news_detection.ablation import ablation_matrices, length_audit, report_markdown, run_ablation
from test_validation import make_data


def test_feature_sets():
    data = make_data()
    matrices, labels = ablation_matrices(data)
    assert list(matrices['length_only']) == ['total_palavras']
    assert all(not col.startswith('contagem_') or col.endswith('_prop') for col in matrices['emotion_only'])
    assert 'total_palavras' not in matrices['emotion_only']
    assert 'total_palavras' in matrices['emotion_plus_length']
    assert len(matrices['means_only'].columns) == 4
    assert len(labels) == len(data)


def test_distribution():
    s = length_audit(make_data())
    assert s['fake']['n'] == s['true']['n'] == 18
    assert s['fake']['min'] > 0
    assert s['true']['p25'] <= s['true']['median'] <= s['true']['p75']


def test_run_ablation_smoke(tmp_path):
    out = run_ablation(make_data(48), cv=2, inner_cv=2, fast=True, n_jobs=1,
                       checkpoint=tmp_path / 'ablation.json')
    assert len(out['folds']) == 2
    assert len(out['summary']) == 4
    assert out['group_exact_duplicate_texts']
    assert (tmp_path / 'ablation.json').exists()
    assert 'length_only' in report_markdown(out)
    assert len(out['paired_f1_differences']['emotion_only_minus_length_only']['by_fold']) == 2
