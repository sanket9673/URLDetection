"""
Unit and Integration Tests for Agentic RAG Orchestrator and Incident Report Synthesizer.
"""

import json
from unittest.mock import MagicMock, patch
import pytest

from src.explainability.agentic_rag import (
    AgenticRAGOrchestrator,
    IncidentReport,
    get_agentic_rag_orchestrator
)


class TestAgenticRAGOrchestrator:
    """Test suite for Agentic RAG Orchestration and Gemini Integration."""

    @pytest.fixture(scope="module")
    def orchestrator(self) -> AgenticRAGOrchestrator:
        return get_agentic_rag_orchestrator(api_key=None)

    def test_orchestrator_initialization(self, orchestrator: AgenticRAGOrchestrator):
        """Verifies components are wired properly."""
        assert orchestrator.shap_engine is not None
        assert orchestrator.graph_rag_engine is not None
        assert orchestrator.mitre_mapper is not None

    def test_build_grounded_context(self, orchestrator: AgenticRAGOrchestrator):
        """Verifies unification of SHAP, Graph RAG, and MITRE data."""
        test_url = "http://paypal-verification-secure-login-account89.com/login.php"
        context = orchestrator.build_grounded_context(test_url)

        assert "query_url" in context
        assert context["query_url"] == test_url
        assert "shap_context" in context
        assert "graph_context" in context
        assert "mitre_category_techniques" in context

        # Check SHAP content
        assert "top_contributing_features" in context["shap_context"]
        assert len(context["shap_context"]["top_contributing_features"]) == 5

        # Check Graph RAG content
        assert "structural_intelligence" in context["graph_context"]
        assert "topological_evidence_trail" in context["graph_context"]

    def test_prompt_construction(self, orchestrator: AgenticRAGOrchestrator):
        """Verifies prompt adheres to grounding constraints."""
        test_url = "http://x901a-adversary-c2-beacon.xyz/stage2/payload.exe"
        context = orchestrator.build_grounded_context(test_url)
        prompt = orchestrator._construct_prompt(context)

        assert "EVIDENCE PAYLOAD" in prompt
        assert "TOP-5 LEXICAL SHAP FEATURE CONTRIBUTIONS" in prompt
        assert "TOPOLOGICAL INFRASTRUCTURE METRICS" in prompt
        assert "STRICT OPERATIONAL GROUNDING CONSTRAINTS" in prompt
        assert "ZERO-DAY STATUS" in prompt
        assert "Executive Threat Summary" in prompt
        assert "Actionable Mitigation" in prompt

    def test_deterministic_synthesis(self, orchestrator: AgenticRAGOrchestrator):
        """Verifies deterministic synthesis when API is offline."""
        test_url = "http://paypal-verification-secure-login-account89.com/login.php"
        report = orchestrator.orchestrate(test_url, api_key=None)

        assert isinstance(report, IncidentReport)
        assert report.query_url == test_url
        assert report.verdict in ["benign", "phishing", "malware", "defacement"]
        assert report.severity_level in ["CRITICAL", "HIGH", "MEDIUM", "LOW", "BENIGN"]
        assert report.generated_by == "Deterministic Grounded Engine"

        # Check the 4 sections
        assert len(report.executive_summary) > 20
        assert len(report.lexical_analysis) > 20
        assert len(report.topological_context) > 20
        assert len(report.remediation_playbook) >= 1
        assert "Enterprise Threat Incident Report" in report.full_markdown_report

    def test_mock_gemini_api_execution(self, orchestrator: AgenticRAGOrchestrator):
        """Tests LLM execution, response parsing, and validation using mock Gemini API."""
        mock_llm_response = """
### 1. Executive Threat Summary & Verdict Confidence
The investigated target is classified as PHISHING with 98.40% confidence. Severity is assessed at HIGH due to credential harvesting patterns.

### 2. Lexical Feature & SHAP Attribution Analysis
The primary driver is `suspicious_keyword_count` (value: 5) contributing +0.7674 SHAP towards phishing, accompanied by deep path token length.

### 3. Topological Infrastructure & Graph Context
The domain is unobserved (zero-day status), relying on parent TLD `.com` with 93,583 indexed domains and an empirical baseline abuse rate of 21.4%.

### 4. Actionable Mitigation & Remediation Playbook
1. Block domain at perimeter Secure Web Gateway (SWG).
2. Quarantine inbound emails containing the URL signature (MITRE T1566.002).
3. Invalidate active user sessions and reset credentials.
"""
        with patch.object(orchestrator, "_call_gemini_api", return_value=mock_llm_response):
            report = orchestrator.orchestrate(
                "http://paypal-verification-secure-login-account89.com/login.php",
                api_key="mock-api-key"
            )

            assert isinstance(report, IncidentReport)
            assert "Google Gemini API" in report.generated_by
            assert "PHISHING" in report.executive_summary
            assert "suspicious_keyword_count" in report.lexical_analysis
            assert ".com" in report.topological_context
            assert len(report.remediation_playbook) >= 3

    def test_json_schema_serialization(self, orchestrator: AgenticRAGOrchestrator):
        """Validates JSON schema serialization of IncidentReport."""
        report = orchestrator.orchestrate("https://www.google.com")
        report_dict = report.to_dict()

        json_str = json.dumps(report_dict)
        assert len(json_str) > 0

        # Required fields
        for field_name in [
            "query_url", "verdict", "confidence", "severity_level",
            "is_zero_day", "executive_summary", "lexical_analysis",
            "topological_context", "remediation_playbook",
            "mitre_techniques_involved", "grounded_facts",
            "full_markdown_report", "generated_by",
            "generation_time_ms", "timestamp"
        ]:
            assert field_name in report_dict
