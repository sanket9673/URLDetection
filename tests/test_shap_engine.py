"""
Unit and Integration Tests for SHAP Attribution Engine and MITRE ATT&CK Mapper.
"""

import json
import pytest
import numpy as np

from src.explainability.mitre_mapper import (
    MitreMapper,
    MitreTechnique,
    get_mitre_mapper,
    MITRE_TECHNIQUE_CATALOG,
    THREAT_CATEGORY_MITRE_MAP
)
from src.explainability.shap_engine import (
    ShapEngine,
    get_shap_engine,
    CLASS_NAMES
)


class TestMitreMapper:
    """Test suite for MITRE ATT&CK Knowledge Base mapping."""

    @pytest.fixture
    def mapper(self) -> MitreMapper:
        return get_mitre_mapper()

    def test_catalog_initialization(self, mapper: MitreMapper):
        """Verifies catalog is populated with valid MitreTechnique instances."""
        assert len(mapper._catalog) >= 10
        technique = mapper.get_technique_by_id("T1566.002")
        assert technique is not None
        assert isinstance(technique, MitreTechnique)
        assert technique.technique_id == "T1566.002"
        assert technique.technique_name == "Phishing: Spearphishing Link"
        assert technique.tactic == "Initial Access"
        assert technique.tactic_id == "TA0001"
        assert "attack.mitre.org" in technique.url_reference

    def test_threat_category_mappings(self, mapper: MitreMapper):
        """Verifies mapping of threat categories to MITRE techniques."""
        phishing_techs = mapper.get_techniques_for_threat_category("phishing")
        assert len(phishing_techs) > 0
        tech_ids = [t.technique_id for t in phishing_techs]
        assert "T1566.002" in tech_ids

        malware_techs = mapper.get_techniques_for_threat_category("malware")
        assert len(malware_techs) > 0
        malware_ids = [t.technique_id for t in malware_techs]
        assert "T1204.002" in malware_ids

    def test_feature_to_technique_mapping(self, mapper: MitreMapper):
        """Tests individual feature mapping to MITRE techniques."""
        # Keyword feature -> Spearphishing
        kw_techs = mapper.map_feature_to_techniques("suspicious_keyword_count", feature_value=3)
        assert len(kw_techs) > 0
        assert any(t.technique_id == "T1566.002" for t in kw_techs)

        # Executable flag -> User Execution Malicious File
        exe_techs = mapper.map_feature_to_techniques("has_exe_or_zip", feature_value=1)
        assert len(exe_techs) > 0
        assert any(t.technique_id == "T1204.002" for t in exe_techs)

        # IP in domain -> Proxy / Direct IP Routing
        ip_techs = mapper.map_feature_to_techniques("contains_ip", feature_value=1)
        assert len(ip_techs) > 0
        assert any(t.technique_id == "T1090.003" for t in ip_techs)

        # Entropy -> DGA
        entropy_techs = mapper.map_feature_to_techniques("entropy", feature_value=4.5)
        assert len(entropy_techs) > 0
        assert any(t.technique_id == "T1568.002" for t in entropy_techs)

    def test_enrich_feature_attribution(self, mapper: MitreMapper):
        """Verifies enrichment of SHAP feature attribution tuple."""
        enriched = mapper.enrich_feature_attribution(
            feature_name="suspicious_keyword_count",
            feature_value=2,
            shap_value=0.4567,
            threat_category="phishing"
        )
        assert enriched["feature_name"] == "suspicious_keyword_count"
        assert enriched["feature_value"] == 2
        assert enriched["shap_value"] == 0.4567
        assert enriched["impact_direction"] == "POSITIVE"
        assert len(enriched["mitre_techniques"]) > 0
        assert enriched["primary_mitre_technique"]["technique_id"] == "T1566.002"


class TestShapEngine:
    """Test suite for the SHAP Attribution Engine."""

    @pytest.fixture(scope="module")
    def engine(self) -> ShapEngine:
        return get_shap_engine("models/lightgbm_model.pkl")

    def test_engine_initialization(self, engine: ShapEngine):
        """Verifies model loading and feature configuration."""
        assert engine._model is not None
        assert len(engine.feature_names) == 118
        assert engine._num_classes == 4

    def test_feature_extraction(self, engine: ShapEngine):
        """Tests URL feature extraction and alignment."""
        test_url = "http://paypal-security-verification.com/login.php?id=992"
        df_feat = engine.extract_features_from_url(test_url)
        assert df_feat.shape == (1, 118)
        assert "url_length" in df_feat.columns
        assert "entropy" in df_feat.columns
        assert df_feat["suspicious_keyword_count"].values[0] >= 1

    def test_explain_url_phishing(self, engine: ShapEngine):
        """Verifies structured Top-5 SHAP output for a phishing URL."""
        phishing_url = "http://paypal-verification-secure-login-account89.com/login.php"
        result = engine.explain_url(phishing_url, top_k=5)

        # JSON Serialization check
        json_str = json.dumps(result)
        assert len(json_str) > 0

        # Validate Schema
        assert "url" in result
        assert result["url"] == phishing_url
        assert "prediction" in result
        assert "predicted_class" in result["prediction"]
        assert "predicted_class_name" in result["prediction"]
        assert "confidence_percent" in result["prediction"]
        assert "all_class_probabilities" in result["prediction"]

        assert "attribution_target" in result
        assert "base_value" in result["attribution_target"]
        assert "total_shap_sum" in result["attribution_target"]

        # Validate Top-5 Features
        top_features = result["top_contributing_features"]
        assert len(top_features) == 5
        for i, feat in enumerate(top_features, start=1):
            assert feat["rank"] == i
            assert "feature_name" in feat
            assert "feature_value" in feat
            assert "shap_value" in feat
            assert feat["impact_direction"] in ["POSITIVE", "NEGATIVE"]
            assert "mitre_techniques" in feat
            assert isinstance(feat["mitre_techniques"], list)

        # Validate MITRE Tactical Summary
        assert "mitre_attack_summary" in result
        assert isinstance(result["mitre_attack_summary"], list)
        assert "metadata" in result
        assert result["metadata"]["top_k"] == 5

    def test_explain_url_malware(self, engine: ShapEngine):
        """Verifies explanation on a malware executable URL."""
        malware_url = "http://192.168.1.100/downloads/ransomware_payload.exe"
        result = engine.explain_url(malware_url, top_k=5)
        top_feat_names = [f["feature_name"] for f in result["top_contributing_features"]]
        
        # Should highlight executable or IP indicator
        assert any(k in top_feat_names for k in ["has_exe_or_zip", "contains_ip", "path_length", "url_length"])

    def test_target_class_override(self, engine: ShapEngine):
        """Verifies target class attribution override."""
        benign_url = "https://www.google.com"
        # Explain with respect to malware (class 3)
        result = engine.explain_url(benign_url, target_class="malware", top_k=5)
        assert result["attribution_target"]["class_index"] == 3
        assert result["attribution_target"]["class_name"] == "malware"
        assert len(result["top_contributing_features"]) == 5

    def test_batch_url_explanation(self, engine: ShapEngine):
        """Verifies batch explanations consistency and speed."""
        batch_urls = [
            "https://www.wikipedia.org",
            "http://secure-banking-alert.com/account/update",
            "http://evil-payload-drop.org/virus.zip"
        ]
        batch_results = engine.explain_batch_urls(batch_urls, top_k=5)
        assert len(batch_results) == 3
        for idx, res in enumerate(batch_results):
            assert res["url"] == batch_urls[idx]
            assert len(res["top_contributing_features"]) == 5
            assert "prediction" in res

    def test_edge_cases(self, engine: ShapEngine):
        """Verifies robustness on empty or atypical inputs."""
        # Empty string
        res_empty = engine.explain_url("", top_k=5)
        assert res_empty is not None
        assert len(res_empty["top_contributing_features"]) == 5

        # Single character URL
        res_single = engine.explain_url("a", top_k=5)
        assert res_single is not None
        assert len(res_single["top_contributing_features"]) == 5
