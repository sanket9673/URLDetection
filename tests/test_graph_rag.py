"""
Unit and Integration Tests for GNN Subgraph Extractor and Graph RAG Context Synthesizer.
"""

from concurrent.futures import ThreadPoolExecutor
import json
import pytest

from src.explainability.graph_rag import (
    GraphRAGEngine,
    GraphRAGContext,
    StructuralIntelligence,
    get_graph_rag_engine
)


class TestGraphRAGEngine:
    """Test suite for the Graph RAG Engine and Subgraph Extractor."""

    @pytest.fixture(scope="module")
    def engine(self) -> GraphRAGEngine:
        return get_graph_rag_engine(
            graph_path="models/gnn_graph_data.pt",
            mappings_path="models/gnn_mappings.pkl"
        )

    def test_engine_initialization(self, engine: GraphRAGEngine):
        """Verifies graph topology and lookup tables are properly indexed."""
        assert engine._graph_data is not None
        assert len(engine._domain_mapping) > 100000
        assert len(engine._tld_mapping) > 500
        assert len(engine._domain_to_urls) > 0
        assert len(engine._domain_to_tld) > 0
        assert len(engine._tld_stats) > 0

    def test_known_domain_subgraph_extraction(self, engine: GraphRAGEngine):
        """Tests 1-hop and 2-hop extraction for a persistent/known domain."""
        # Pick a domain that exists in the domain_mapping
        sample_domain = next(iter(engine._domain_mapping.keys()))
        sample_url = f"http://{sample_domain}/test/path"

        context = engine.extract_context(sample_url, max_neighbor_urls=5)

        assert isinstance(context, GraphRAGContext)
        assert context.resolved_domain == sample_domain
        assert context.is_zero_day is False

        # Topology structure
        topo = context.subgraph_topology
        assert topo["num_domain_nodes"] == 1
        assert topo["num_tld_nodes"] == 1
        assert topo["num_url_nodes"] >= 1
        assert topo["total_nodes"] == topo["num_domain_nodes"] + topo["num_tld_nodes"] + topo["num_url_nodes"]
        assert topo["total_edges"] >= 2  # URL->Domain and Domain->TLD

        # Structural Intelligence
        si = context.structural_intelligence
        assert isinstance(si, StructuralIntelligence)
        assert 0.0 <= si.neighbor_threat_density <= 1.0
        assert 0.0 <= si.tld_historical_risk_score <= 1.0
        assert si.total_domain_urls >= 1
        assert si.infrastructure_risk_level in ["CRITICAL", "HIGH", "MEDIUM", "LOW", "BENIGN"]

        # Evidence Trail
        assert len(context.topological_evidence_trail) >= 2
        assert any(sample_domain in e for e in context.topological_evidence_trail)

    def test_zero_day_cold_start_domain(self, engine: GraphRAGEngine):
        """Tests fallback and inductive prior reasoning for an unseen zero-day domain."""
        unseen_url = "http://totally-new-unseen-threat-domain-999.xyz/payload.exe"
        context = engine.extract_context(unseen_url, max_neighbor_urls=5)

        assert context.is_zero_day is True
        assert context.resolved_domain == "totally-new-unseen-threat-domain-999.xyz"
        assert context.resolved_tld == "xyz"

        # Topology structure
        topo = context.subgraph_topology
        assert topo["num_domain_nodes"] == 1
        assert topo["num_tld_nodes"] == 1
        assert topo["num_url_nodes"] == 1

        # Structural Intelligence
        si = context.structural_intelligence
        assert 0.0 <= si.neighbor_threat_density <= 1.0
        assert 0.0 <= si.tld_historical_risk_score <= 1.0

        # Evidence Trail
        assert len(context.topological_evidence_trail) >= 2
        assert any("ZERO-DAY INDUCTIVE INFERENCE" in e for e in context.topological_evidence_trail)

    def test_unseen_tld_handling(self, engine: GraphRAGEngine):
        """Tests handling of completely unknown TLD strings via n-gram hashing."""
        custom_url = "http://cyber-threat-actor.unknowncustomtld/phish"
        context = engine.extract_context(custom_url)

        assert context.is_zero_day is True
        assert context.resolved_tld == "unknowncustomtld"
        assert 0.0 <= context.structural_intelligence.tld_historical_risk_score <= 1.0
        assert any("unknowncustomtld" in e for e in context.topological_evidence_trail)

    def test_json_schema_serialization(self, engine: GraphRAGEngine):
        """Validates exact JSON schema serialization of Graph RAG Context."""
        context_dict = engine.extract_context_json("http://paypal-verification-secure.com/login")

        # Must be valid JSON
        json_str = json.dumps(context_dict)
        assert len(json_str) > 0

        # Verify Top-Level Keys
        required_keys = [
            "query", "resolved_domain", "resolved_tld", "is_zero_day",
            "subgraph_topology", "structural_intelligence",
            "topological_evidence_trail", "metadata"
        ]
        for k in required_keys:
            assert k in context_dict

        # Verify Structural Intelligence Keys
        si = context_dict["structural_intelligence"]
        for si_k in [
            "neighbor_threat_density", "neighbor_class_breakdown",
            "total_domain_urls", "tld_historical_risk_score",
            "tld_total_domains", "tld_total_urls",
            "infrastructure_risk_level", "degree_centrality"
        ]:
            assert si_k in si

        # Verify Metadata Keys
        meta = context_dict["metadata"]
        assert "engine" in meta
        assert "extraction_time_ms" in meta
        assert "timestamp" in meta

    def test_concurrent_extraction(self, engine: GraphRAGEngine):
        """Tests thread-safe concurrent subgraph extractions under multi-user load."""
        test_queries = [
            "https://www.google.com/search?q=security",
            "http://paypal-verification-secure-login-account89.com/login.php",
            "http://zero-day-malware-drop-test-88.biz/download.zip",
            "http://unseen-phish-domain-331.club/verify",
            "https://www.wikipedia.org/wiki/Computer_security"
        ]

        def worker(q: str):
            return engine.extract_context(q)

        with ThreadPoolExecutor(max_workers=5) as executor:
            results = list(executor.map(worker, test_queries * 4))

        assert len(results) == 20
        for res in results:
            assert isinstance(res, GraphRAGContext)
            assert res.metadata["extraction_time_ms"] >= 0.0
