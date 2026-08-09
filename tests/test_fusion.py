import pytest
import numpy as np

def test_alpha_blending():
    # 4-class prediction probabilities [Benign, Phishing, Defacement, Malware]
    # For two URLs
    P_feature = np.array([
        [0.70, 0.10, 0.10, 0.10],
        [0.05, 0.90, 0.02, 0.03]
    ])
    P_graph = np.array([
        [0.80, 0.05, 0.10, 0.05],
        [0.15, 0.70, 0.10, 0.05]
    ])
    
    alpha = 0.7
    beta = 1.0 - alpha
    
    # Blended output
    P_final = alpha * P_feature + beta * P_graph
    
    # Verify values for first URL
    expected_url0 = 0.7 * P_feature[0] + 0.3 * P_graph[0]
    assert np.allclose(P_final[0], expected_url0)
    
    # Verify values for second URL
    expected_url1 = 0.7 * P_feature[1] + 0.3 * P_graph[1]
    assert np.allclose(P_final[1], expected_url1)
    
    # Verify normalization sum(P) = 1.0 for all rows
    row_sums = np.sum(P_final, axis=1)
    assert np.allclose(row_sums, 1.0)
