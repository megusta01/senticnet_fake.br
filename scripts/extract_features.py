"""Run from the project root after obtaining the corpus and SenticNet data."""
import argparse
import importlib.util
from pathlib import Path

import pandas as pd
from fake_news_detection.features import extract_emotional_features


def load_lexicon(path: Path):
    if not path.is_file():
        raise FileNotFoundError(f'SenticNet dictionary not found: {path}')
    spec = importlib.util.spec_from_file_location('local_senticnet_data', path)
    if spec is None or spec.loader is None:
        raise ValueError(f'Cannot import SenticNet dictionary: {path}')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.senticnet


def main():
    parser = argparse.ArgumentParser(description='Extract emotional features from Fake.Br news.')
    parser.add_argument('--input', type=Path, default=Path('data/pre-processed.csv'))
    parser.add_argument('--senticnet', type=Path, default=Path('data/senticnet_data.py'))
    parser.add_argument('--output', type=Path, default=Path('data/analise_emocional_completa.csv'))
    args = parser.parse_args()
    frame = pd.read_csv(args.input)
    if 'preprocessed_news' not in frame.columns:
        raise ValueError('CSV must contain a preprocessed_news column.')
    lexicon = load_lexicon(args.senticnet)
    features = frame['preprocessed_news'].apply(lambda text: extract_emotional_features(text, lexicon))
    result = pd.concat([frame.reset_index(drop=True), pd.DataFrame(features.tolist())], axis=1)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print(f'Saved {len(result)} articles to {args.output}')


if __name__ == '__main__':
    main()
