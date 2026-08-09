import pytest
import os
import pandas as pd
import torch
from unittest.mock import patch, mock_open
from src.graph.gnn_train import prepare_hetero_graph, HeteroGraphSAGE

@patch('torch.save')
@patch('pickle.dump')
@patch('src.graph.gnn_train.open', new_callable=mock_open)
def test_gnn_graph_and_model(mock_file, mock_pickle, mock_torch_save, tmp_path):
    # Dummy dataset containing 40 features
    # Ensure sufficient members per class for stratified train/val/test splits (e.g. 20 rows)
    dummy_data = {
        'url': [f'http://test{i % 2 + 1}.com/path{i}' for i in range(20)],
        'type': ['benign', 'phishing'] * 10,
        'target': [0, 2] * 10
    }
    
    # Fill in all 40 required feature columns
    feature_cols = [
        'url_length', 'domain_length', 'path_length', 'subdomain_count',
        'num_dots', 'num_digits', 'num_hyphens', 'num_special_chars',
        'https_flag', 'contains_ip', 'num_underscores', 'num_equals',
        'num_ampersands', 'num_percent', 'num_semicolons', 'num_tilde',
        'num_plus', 'num_asterisk', 'num_hash', 'vowel_count',
        'consonant_count', 'letter_count', 'has_port', 'entropy',
        'domain_entropy', 'unique_char_ratio', 'digit_ratio',
        'special_char_ratio', 'query_param_count', 'longest_token_length',
        'vowel_ratio', 'consonant_ratio', 'letter_ratio',
        'suspicious_keyword_count', 'contains_at', 'double_slash_count',
        'is_shortened', 'has_exe_or_zip', 'suspicious_tld', 'multiple_subdomains'
    ]
    for col in feature_cols:
        dummy_data[col] = [1.0 + (i * 0.1) for i in range(20)]
        
    df = pd.DataFrame(dummy_data)
    parquet_path = os.path.join(tmp_path, "dummy_feats.parquet")
    df.to_parquet(parquet_path, index=False)
    
    # 1. Test Graph construction
    data, le, df_out = prepare_hetero_graph(df_path=parquet_path, target_col='target')
    
    # Verify nodes
    assert data['url'].x.shape[0] == 20
    # Two unique domains: test1.com, test2.com
    assert data['domain'].x.shape[0] == 2
    # One unique TLD: com
    assert data['tld'].x.shape[0] == 1
    
    # Verify node ID mappings completeness
    assert mock_pickle.called
    assert mock_torch_save.called
    
    # 2. Test Model prediction shape
    model = HeteroGraphSAGE(hidden_channels=8, out_channels=4, metadata=data.metadata())
    model.eval()
    with torch.no_grad():
        out_probs = model(data.x_dict, data.edge_index_dict)
        
    # Verify predictions shape [N, 4]
    assert out_probs.shape == (20, 4)
    # Check row sums are normalized
    row_sums = out_probs.sum(dim=1)
    assert torch.allclose(row_sums, torch.ones_like(row_sums))
