"""
Integration Test for Streamlit Dashboard Components and Assets.
"""

import pytest
import numpy as np
import pandas as pd


def test_dashboard_asset_loading():
    """Verifies all assets and explainability engines loaded by dashboard are intact."""
    from src.feature_engineering.feature_builder import FeatureBuilder
    from src.explainability.shap_engine import get_shap_engine
    from src.explainability.graph_rag import get_graph_rag_engine
    from src.explainability.mitre_mapper import get_mitre_mapper
    from src.explainability.agentic_rag import get_agentic_rag_orchestrator
    from src.prediction_guard import check_whitelist, apply_prediction_guard

    builder = FeatureBuilder(raw_data_path="", output_path="")
    assert builder is not None

    shap_engine = get_shap_engine("models/lightgbm_model.pkl")
    assert shap_engine is not None

    graph_rag_engine = get_graph_rag_engine("models/gnn_graph_data.pt", "models/gnn_mappings.pkl")
    assert graph_rag_engine is not None

    mitre_mapper = get_mitre_mapper()
    assert mitre_mapper is not None

    orchestrator = get_agentic_rag_orchestrator(api_key=None)
    assert orchestrator is not None

    # Test Whitelist check
    is_wl, p_wl = check_whitelist("https://www.google.com")
    assert is_wl is True
    assert p_wl is not None

    # Test Tab 2 end-to-end payload synthesis
    test_url = "http://paypal-verification-secure-login-account89.com/login.php"
    shap_res = shap_engine.explain_url(test_url, top_k=5)
    assert "top_contributing_features" in shap_res

    graph_rag_res = graph_rag_engine.extract_context_json(test_url, max_neighbor_urls=10)
    assert "structural_intelligence" in graph_rag_res

    incident_report = orchestrator.orchestrate(test_url)
    assert incident_report.verdict == "phishing"
    assert len(incident_report.remediation_playbook) >= 1
