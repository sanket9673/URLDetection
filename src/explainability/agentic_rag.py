"""
Agentic RAG Orchestrator & Google Gemini API Integration Module.

Synthesizes symbolic SHAP feature attributions, GNN topological subgraphs,
and MITRE ATT&CK tactical techniques into grounded, zero-hallucination,
4-part enterprise incident response playbooks using Google Gemini.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
import json
import logging
import os
import re
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

import requests

from src.explainability.graph_rag import GraphRAGEngine, get_graph_rag_engine
from src.explainability.mitre_mapper import MitreMapper, get_mitre_mapper
from src.explainability.shap_engine import ShapEngine, get_shap_engine

try:
    from src.logger_config import get_logger
    logger = get_logger(__name__)
except ImportError:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    logger = logging.getLogger(__name__)


# Standard Default Gemini Models
DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"
FALLBACK_GEMINI_MODEL = "gemini-1.5-flash"


@dataclass
class IncidentReport:
    """Standardized 4-Part Enterprise Cyber-Threat Incident Response Playbook."""

    query_url: str
    verdict: str
    confidence: float
    severity_level: str  # 'CRITICAL', 'HIGH', 'MEDIUM', 'LOW', 'BENIGN'
    is_zero_day: bool
    executive_summary: str
    lexical_analysis: str
    topological_context: str
    remediation_playbook: List[str]
    mitre_techniques_involved: List[Dict[str, Any]]
    grounded_facts: Dict[str, Any]
    full_markdown_report: str
    generated_by: str  # 'Gemini API' or 'Deterministic Grounded Engine'
    generation_time_ms: float
    timestamp: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class AgenticRAGOrchestrator:
    """
    Central orchestration engine that merges SHAP lexical attributions,
    GNN topological subgraphs, and MITRE ATT&CK techniques with Google Gemini.
    """

    def __init__(
        self,
        shap_engine: Optional[ShapEngine] = None,
        graph_rag_engine: Optional[GraphRAGEngine] = None,
        mitre_mapper: Optional[MitreMapper] = None,
        api_key: Optional[str] = None,
        model_name: str = DEFAULT_GEMINI_MODEL
    ) -> None:
        """
        Initializes the Agentic RAG Orchestrator.
        
        Args:
            shap_engine: Optional pre-configured ShapEngine.
            graph_rag_engine: Optional pre-configured GraphRAGEngine.
            mitre_mapper: Optional pre-configured MitreMapper.
            api_key: Optional Google Gemini API key (defaults to GEMINI_API_KEY env var).
            model_name: Google Gemini model name to use.
        """
        self.shap_engine = shap_engine or get_shap_engine()
        self.graph_rag_engine = graph_rag_engine or get_graph_rag_engine()
        self.mitre_mapper = mitre_mapper or get_mitre_mapper()
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        self.model_name = model_name

        logger.info(
            f"Initialized AgenticRAGOrchestrator (model: {self.model_name}, "
            f"API Key present: {bool(self.api_key)})"
        )

    def build_grounded_context(self, url: str) -> Dict[str, Any]:
        """
        Extracts and unifies the three symbolic data sources for the query URL.
        
        Args:
            url: The query URL string.
            
        Returns:
            Dictionary containing SHAP attributions, Graph RAG context, and MITRE mappings.
        """
        start_t = time.perf_counter()
        
        # 1. Lexical SHAP Attribution
        shap_context = self.shap_engine.explain_url(url, top_k=5)
        
        # 2. Topological Graph RAG Context
        graph_context = self.graph_rag_engine.extract_context_json(url, max_neighbor_urls=10)
        
        # 3. MITRE ATT&CK Technique Mapping
        predicted_class_name = shap_context["prediction"]["predicted_class_name"]
        category_mitre = self.mitre_mapper.get_techniques_for_threat_category(predicted_class_name)
        category_mitre_dicts = [t.to_dict() for t in category_mitre]
        
        elapsed_ms = (time.perf_counter() - start_t) * 1000.0
        
        return {
            "query_url": url,
            "shap_context": shap_context,
            "graph_context": graph_context,
            "mitre_category_techniques": category_mitre_dicts,
            "extraction_latency_ms": round(elapsed_ms, 2)
        }

    def _construct_prompt(self, context: Dict[str, Any]) -> str:
        """
        Builds a dense, grounded, constraint-bounded prompt for the Gemini LLM.
        """
        shap_ctx = context["shap_context"]
        graph_ctx = context["graph_context"]
        pred = shap_ctx["prediction"]
        struct_intel = graph_ctx["structural_intelligence"]
        
        top_features = shap_ctx["top_contributing_features"]
        evidence_trail = graph_ctx["topological_evidence_trail"]
        mitre_summary = shap_ctx.get("mitre_attack_summary", [])
        
        prompt = f"""You are the Lead Cyber-Threat Intelligence (CTI) AI for an enterprise Security Operations Center (SOC).
Analyze the following multi-source evidence payload for the investigated URL.

EVIDENCE PAYLOAD:
--------------------------------------------------------------------------------
1. QUERY URL: {context['query_url']}
2. VERDICT: {pred['predicted_class_name'].upper()} (Confidence: {pred['confidence_percent']}%)
   - Class Probabilities: {json.dumps(pred['all_class_probabilities'])}
3. ZERO-DAY STATUS: {'YES (Unseen Domain in Training Graph)' if graph_ctx['is_zero_day'] else 'NO (Known Indexed Domain)'}
4. TOP-5 LEXICAL SHAP FEATURE CONTRIBUTIONS:
{json.dumps(top_features, indent=2)}

5. TOPOLOGICAL INFRASTRUCTURE METRICS (Graph RAG):
   - Resolved Domain: {graph_ctx['resolved_domain']}
   - Parent TLD: .{graph_ctx['resolved_tld']} (Historical Abuse Risk: {struct_intel['tld_historical_risk_score'] * 100.0:.1f}%, Total Domains: {struct_intel['tld_total_domains']:,})
   - Domain Neighbor Threat Density: {struct_intel['neighbor_threat_density'] * 100.0:.1f}% ({json.dumps(struct_intel['neighbor_class_breakdown'])})
   - Infrastructure Risk Tier: {struct_intel['infrastructure_risk_level']}
   - Topological Evidence Trail:
{chr(10).join('     * ' + e for e in evidence_trail)}

6. ASSOCIATED MITRE ATT&CK TACTICS & TECHNIQUES:
{json.dumps(mitre_summary, indent=2)}
--------------------------------------------------------------------------------

STRICT OPERATIONAL GROUNDING CONSTRAINTS (ZERO HALLUCINATION):
1. Rely EXCLUSIVELY on the exact feature names, SHAP values, threat densities, TLD abuse rates, and MITRE Technique IDs provided above.
2. DO NOT invent fictitious IP addresses, external WHOIS dates, or unlisted CVEs.
3. Every claim about the URL's lexical risk MUST reference the specific SHAP feature name and value from the payload.
4. Every claim about infrastructure risk MUST reference the exact neighbor threat density or TLD risk score.

RESPONSE FORMAT REQUIREMENTS:
Output your final assessment strictly divided into these exact 4 Markdown sections:

### 1. Executive Threat Summary & Verdict Confidence
[State the exact verdict, confidence score, severity rating (CRITICAL/HIGH/MEDIUM/LOW/BENIGN), zero-day status, and brief operational risk impact.]

### 2. Lexical Feature & SHAP Attribution Analysis
[Analyze the top lexical indicators driving the prediction, explicitly citing the top SHAP features, their values, and whether they pushed towards or against the classification.]

### 3. Topological Infrastructure & Graph Context
[Analyze the domain and TLD neighborhood from the Graph RAG evidence trail, citing exact neighbor threat density, historical TLD risk, and infrastructure co-occurrence.]

### 4. Actionable Mitigation & Remediation Playbook
[Provide 3-5 concrete, step-by-step containment and remediation actions aligned with the cited MITRE ATT&CK techniques (e.g. perimeter block rules, proxy inspection, EDR quarantine).]
"""
        return prompt

    def _call_gemini_api(self, prompt: str, api_key: str) -> Optional[str]:
        """
        Calls Google Gemini API via direct REST with exponential backoff.
        
        Args:
            prompt: Formatted system prompt.
            api_key: Valid Google Gemini API Key.
            
        Returns:
            Generated response string or None if API call failed.
        """
        # Endpoint options for Google Gemini models
        models_to_try = [self.model_name, DEFAULT_GEMINI_MODEL, FALLBACK_GEMINI_MODEL]
        
        # Deduplicate models
        seen = set()
        models = [m for m in models_to_try if not (m in seen or seen.add(m))]

        for model in models:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
            headers = {"Content-Type": "application/json"}
            payload = {
                "contents": [
                    {
                        "parts": [{"text": prompt}]
                    }
                ],
                "generationConfig": {
                    "temperature": 0.2,  # Low temperature for strict analytical grounding
                    "topP": 0.8,
                    "maxOutputTokens": 2048
                }
            }

            for attempt in range(1, 4):
                try:
                    logger.info(f"Sending prompt to Gemini API ({model}, attempt {attempt}/3)...")
                    response = requests.post(url, headers=headers, json=payload, timeout=25.0)
                    
                    if response.status_code == 200:
                        data = response.json()
                        candidates = data.get("candidates", [])
                        if candidates:
                            text = candidates[0].get("content", {}).get("parts", [{}])[0].get("text", "")
                            if text:
                                logger.info(f"Successfully generated response via Gemini API ({model}).")
                                return text
                    elif response.status_code == 429:
                        logger.warning(f"Rate limited by Gemini API (429). Retrying in {attempt * 2}s...")
                        time.sleep(attempt * 2.0)
                    else:
                        logger.warning(f"Gemini API returned HTTP {response.status_code}: {response.text[:200]}")
                        break
                except Exception as e:
                    logger.warning(f"Gemini API network error on attempt {attempt}: {str(e)}")
                    time.sleep(attempt * 1.5)

        logger.error("All Gemini API attempts failed. Falling back to deterministic synthesizer.")
        return None

    def _synthesize_deterministic_report(
        self,
        context: Dict[str, Any],
        elapsed_ms: float
    ) -> IncidentReport:
        """
        Generates a 100% grounded, deterministic incident response playbook
        when Gemini API is offline or without an API key.
        """
        shap_ctx = context["shap_context"]
        graph_ctx = context["graph_context"]
        pred = shap_ctx["prediction"]
        struct_intel = graph_ctx["structural_intelligence"]
        
        verdict = pred["predicted_class_name"]
        confidence = pred["confidence_percent"]
        is_zero_day = graph_ctx["is_zero_day"]
        top_features = shap_ctx["top_contributing_features"]
        mitre_summary = shap_ctx.get("mitre_attack_summary", [])
        
        # Determine Severity Level
        if verdict == "malware" or struct_intel["infrastructure_risk_level"] == "CRITICAL":
            severity = "CRITICAL"
        elif verdict == "phishing" or struct_intel["infrastructure_risk_level"] == "HIGH":
            severity = "HIGH"
        elif verdict == "defacement" or struct_intel["infrastructure_risk_level"] == "MEDIUM":
            severity = "MEDIUM"
        elif verdict == "benign":
            severity = "BENIGN"
        else:
            severity = "LOW"

        # 1. Executive Summary
        exec_summary = (
            f"The investigated target '{context['query_url']}' has been classified as **{verdict.upper()}** "
            f"with a model confidence of **{confidence:.2f}%** (Severity: **{severity}**). "
            f"{'The destination represents an unobserved zero-day domain exhibiting cold-start inductive risk.' if is_zero_day else 'The destination resides on a historically indexed domain within the enterprise topology.'}"
        )

        # 2. Lexical Feature Analysis
        lexical_lines = [
            f"The LightGBM TreeSHAP attribution identified the following primary lexical indicators:"
        ]
        for f in top_features:
            sign = "pushed toward risk" if f["impact_direction"] == "POSITIVE" else "reduced threat score"
            lexical_lines.append(
                f"- **{f['feature_name']}** (Value: `{f['feature_value']}`): Contributed **{f['shap_value']:+.4f}** SHAP value ({sign})."
            )
        lexical_analysis = "\n".join(lexical_lines)

        # 3. Topological Infrastructure Context
        dom_history_str = f"Associated with {struct_intel['total_domain_urls']} historical URLs in graph"
        dom_status_str = "Zero-Day Unseen Domain" if is_zero_day else dom_history_str
        
        topological_lines = [
            f"Graph RAG ego-network analysis across the heterogeneous PyG topology revealed:",
            f"- **Resolved Domain**: `{graph_ctx['resolved_domain']}` ({dom_status_str}).",
            f"- **Neighbor Threat Density**: `{struct_intel['neighbor_threat_density'] * 100.0:.1f}%` of adjacent graph nodes exhibit malicious classification.",
            f"- **Parent TLD Risk**: `.{graph_ctx['resolved_tld']}` has an empirical abuse baseline of `{struct_intel['tld_historical_risk_score'] * 100.0:.1f}%` across `{struct_intel['tld_total_domains']:,}` domains.",
            f"- **Infrastructure Risk Tier**: **{struct_intel['infrastructure_risk_level']}**."
        ]
        topological_context = "\n".join(topological_lines)

        # 4. Remediation Playbook
        remediation_playbook: List[str] = []
        if verdict != "benign":
            remediation_playbook.append(
                f"**Perimeter Gateway Block**: Implement immediate DNS and Secure Web Gateway (SWG) sinkholing for domain `{graph_ctx['resolved_domain']}`."
            )
            if any(t["technique_id"].startswith("T1566") for t in mitre_summary):
                remediation_playbook.append(
                    "**Email Quarantine & Token Revocation**: Invalidate active session cookies and quarantine inbound emails matching this URL pattern (MITRE T1566.002)."
                )
            if any(t["technique_id"].startswith("T1204") for t in mitre_summary) or verdict == "malware":
                remediation_playbook.append(
                    "**Endpoint EDR Sweep**: Initiate an enterprise-wide EDR hash sweep for downloaded binaries/payloads and isolate communicating hosts (MITRE T1204.002)."
                )
            remediation_playbook.append(
                f"**TLD Risk Monitoring**: Apply heightened DPI inspection for traffic destined to high-abuse TLD `.{graph_ctx['resolved_tld']}` (MITRE T1583.001)."
            )
        else:
            remediation_playbook.append(
                "**Standard Telemetry Logging**: No active containment required. Maintain standard proxy access logs."
            )

        # Full Markdown Construction
        markdown_sections = [
            f"# Enterprise Threat Incident Report: {verdict.upper()}",
            f"**Target URL:** `{context['query_url']}`  \n**Verdict:** `{verdict.upper()}` | **Confidence:** `{confidence:.2f}%` | **Severity:** `{severity}`\n",
            f"### 1. Executive Threat Summary & Verdict Confidence\n{exec_summary}\n",
            f"### 2. Lexical Feature & SHAP Attribution Analysis\n{lexical_analysis}\n",
            f"### 3. Topological Infrastructure & Graph Context\n{topological_context}\n",
            f"### 4. Actionable Mitigation & Remediation Playbook\n" + "\n".join(f"{i+1}. {step}" for i, step in enumerate(remediation_playbook))
        ]
        full_markdown = "\n\n".join(markdown_sections)

        return IncidentReport(
            query_url=context["query_url"],
            verdict=verdict,
            confidence=float(confidence),
            severity_level=severity,
            is_zero_day=is_zero_day,
            executive_summary=exec_summary,
            lexical_analysis=lexical_analysis,
            topological_context=topological_context,
            remediation_playbook=remediation_playbook,
            mitre_techniques_involved=mitre_summary,
            grounded_facts={
                "top_features": top_features,
                "neighbor_threat_density": struct_intel["neighbor_threat_density"],
                "tld_historical_risk_score": struct_intel["tld_historical_risk_score"],
                "infrastructure_risk_level": struct_intel["infrastructure_risk_level"]
            },
            full_markdown_report=full_markdown,
            generated_by="Deterministic Grounded Engine",
            generation_time_ms=round(elapsed_ms, 2),
            timestamp=datetime.utcnow().isoformat() + "Z"
        )

    def _parse_and_validate_llm_response(
        self,
        raw_text: str,
        context: Dict[str, Any],
        elapsed_ms: float
    ) -> IncidentReport:
        """
        Parses and validates LLM markdown text into the structured IncidentReport schema.
        """
        shap_ctx = context["shap_context"]
        graph_ctx = context["graph_context"]
        pred = shap_ctx["prediction"]
        struct_intel = graph_ctx["structural_intelligence"]
        
        verdict = pred["predicted_class_name"]
        confidence = pred["confidence_percent"]
        is_zero_day = graph_ctx["is_zero_day"]
        top_features = shap_ctx["top_contributing_features"]
        mitre_summary = shap_ctx.get("mitre_attack_summary", [])
        
        if verdict == "malware" or struct_intel["infrastructure_risk_level"] == "CRITICAL":
            severity = "CRITICAL"
        elif verdict == "phishing" or struct_intel["infrastructure_risk_level"] == "HIGH":
            severity = "HIGH"
        elif verdict == "defacement" or struct_intel["infrastructure_risk_level"] == "MEDIUM":
            severity = "MEDIUM"
        elif verdict == "benign":
            severity = "BENIGN"
        else:
            severity = "LOW"

        # Regex extract sections from LLM output cleanly across header titles
        def extract_section(title_keyword: str) -> str:
            pattern = rf"###\s*\d*\.?\s*.*?(?:{title_keyword})[^\n]*\n(.*?)(?=\n###|\Z)"
            match = re.search(pattern, raw_text, re.DOTALL | re.IGNORECASE)
            return match.group(1).strip() if match else ""

        exec_summary = extract_section("Executive")
        lexical_analysis = extract_section("Lexical")
        topological_context = extract_section("Topological")
        remediation_text = extract_section("Mitigation|Remediation")

        # Parse remediation lines
        remediation_playbook = [
            line.strip().lstrip("0123456789.-* ")
            for line in remediation_text.split("\n")
            if line.strip() and not line.strip().startswith("#")
        ]

        # Fallback if LLM failed to format specific subsections
        if not exec_summary:
            exec_summary = f"URL classified as {verdict.upper()} ({confidence:.2f}% confidence)."
        if not lexical_analysis:
            lexical_analysis = "Lexical indicators analyzed via LightGBM TreeSHAP."
        if not topological_context:
            topological_context = f"Topological risk level: {struct_intel['infrastructure_risk_level']}."
        if not remediation_playbook:
            remediation_playbook = [f"Apply perimeter blocking for domain '{graph_ctx['resolved_domain']}'."]

        return IncidentReport(
            query_url=context["query_url"],
            verdict=verdict,
            confidence=float(confidence),
            severity_level=severity,
            is_zero_day=is_zero_day,
            executive_summary=exec_summary,
            lexical_analysis=lexical_analysis,
            topological_context=topological_context,
            remediation_playbook=remediation_playbook,
            mitre_techniques_involved=mitre_summary,
            grounded_facts={
                "top_features": top_features,
                "neighbor_threat_density": struct_intel["neighbor_threat_density"],
                "tld_historical_risk_score": struct_intel["tld_historical_risk_score"],
                "infrastructure_risk_level": struct_intel["infrastructure_risk_level"]
            },
            full_markdown_report=raw_text,
            generated_by=f"Google Gemini API ({self.model_name})",
            generation_time_ms=round(elapsed_ms, 2),
            timestamp=datetime.utcnow().isoformat() + "Z"
        )

    def orchestrate(
        self,
        url: str,
        api_key: Optional[str] = None
    ) -> IncidentReport:
        """
        Executes end-to-end Agentic RAG analysis for a target URL.
        
        Args:
            url: The URL string to inspect.
            api_key: Optional API key override.
            
        Returns:
            Structured IncidentReport object.
        """
        start_t = time.perf_counter()
        
        # 1. Build Grounded Context from SHAP, Graph RAG, and MITRE
        context = self.build_grounded_context(url)
        
        effective_key = api_key or self.api_key
        
        # 2. Execute with Gemini LLM if API Key is available
        if effective_key:
            prompt = self._construct_prompt(context)
            llm_text = self._call_gemini_api(prompt, effective_key)
            if llm_text:
                elapsed_ms = (time.perf_counter() - start_t) * 1000.0
                return self._parse_and_validate_llm_response(llm_text, context, elapsed_ms)

        # 3. Deterministic Grounded Synthesis Fallback
        elapsed_ms = (time.perf_counter() - start_t) * 1000.0
        return self._synthesize_deterministic_report(context, elapsed_ms)


# Module-level cached instance
_GLOBAL_AGENTIC_RAG: Optional[AgenticRAGOrchestrator] = None
_ORCHESTRATOR_LOCK = threading.Lock()


def get_agentic_rag_orchestrator(
    api_key: Optional[str] = None,
    model_name: str = DEFAULT_GEMINI_MODEL
) -> AgenticRAGOrchestrator:
    """
    Returns a thread-safe singleton instance of the AgenticRAGOrchestrator.
    
    Args:
        api_key: Optional Google Gemini API key.
        model_name: Model identifier string.
        
    Returns:
        Cached AgenticRAGOrchestrator instance.
    """
    global _GLOBAL_AGENTIC_RAG
    if _GLOBAL_AGENTIC_RAG is None:
        with _ORCHESTRATOR_LOCK:
            if _GLOBAL_AGENTIC_RAG is None:
                _GLOBAL_AGENTIC_RAG = AgenticRAGOrchestrator(
                    api_key=api_key,
                    model_name=model_name
                )
    return _GLOBAL_AGENTIC_RAG
