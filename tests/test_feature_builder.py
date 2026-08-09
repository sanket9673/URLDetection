import pytest
import pandas as pd
import numpy as np
from src.feature_engineering.feature_builder import FeatureBuilder
from fix_data import fix_url

def test_feature_builder_entropy():
    # Construct a dummy DataFrame
    df = pd.DataFrame([
        {'url': 'aaaa', 'type': 'benign'},
        {'url': 'ab', 'type': 'benign'}
    ])
    builder = FeatureBuilder(raw_data_path="", output_path="")
    df_cleaned = builder.validate_and_clean(df)
    df_feats = builder.build_features(df_cleaned)
    
    # "aaaa" has 1 unique character, entropy should be 0.0
    assert df_feats.loc[0, 'entropy'] == 0.0
    # "ab" has 2 unique characters with equal probability (0.5), entropy should be 1.0
    assert np.isclose(df_feats.loc[1, 'entropy'], 1.0)

def test_feature_builder_keywords():
    df = pd.DataFrame([
        {'url': 'http://secure-login.com/verify/update', 'type': 'phishing'}
    ])
    builder = FeatureBuilder(raw_data_path="", output_path="")
    df_cleaned = builder.validate_and_clean(df)
    df_feats = builder.build_features(df_cleaned)
    
    # Should flag suspicious keywords ('secure', 'login', 'verify', 'update')
    assert df_feats.loc[0, 'suspicious_keyword_count'] >= 3

def test_feature_builder_normalization():
    # Seed for deterministic tests
    np.random.seed(42)
    benign_row = {'url': 'example.com', 'type': 'benign'}
    fixed = fix_url(benign_row)
    assert fixed.startswith('http')
    
    malicious_row = {'url': 'example.com/malicious', 'type': 'phishing'}
    fixed_malicious = fix_url(malicious_row)
    assert fixed_malicious == 'example.com/malicious'

def test_feature_builder_feature_count():
    df = pd.DataFrame([
        {'url': 'https://google.com/search?q=hello', 'type': 'benign'}
    ])
    builder = FeatureBuilder(raw_data_path="", output_path="")
    df_cleaned = builder.validate_and_clean(df)
    df_feats = builder.build_features(df_cleaned)
    df_encoded = builder.encode_target(df_feats)
    
    feature_cols = [c for c in df_encoded.columns if c not in ['url', 'type', 'target']]
    # Verify we extract exactly 40 features
    assert len(feature_cols) == 40
