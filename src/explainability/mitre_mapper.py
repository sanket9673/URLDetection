"""
MITRE ATT&CK Knowledge Base Mapping Module for URL Threat Intelligence.

Maps threat categories, lexical indicators, and statistical URL features to
formal MITRE ATT&CK Enterprise Tactics, Techniques, and Procedures (TTPs).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Set
import logging

try:
    from src.logger_config import get_logger
    logger = get_logger(__name__)
except ImportError:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MitreTechnique:
    """Represents an atomic MITRE ATT&CK Technique or Sub-technique."""

    technique_id: str
    technique_name: str
    tactic: str
    tactic_id: str
    description: str
    mitigation_guidance: str
    detection_methods: str
    url_reference: str
    related_indicators: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Serializes the technique to a dictionary."""
        return asdict(self)


# Comprehensive MITRE ATT&CK Technique Knowledge Base
MITRE_TECHNIQUE_CATALOG: Dict[str, MitreTechnique] = {
    "T1566.002": MitreTechnique(
        technique_id="T1566.002",
        technique_name="Phishing: Spearphishing Link",
        tactic="Initial Access",
        tactic_id="TA0001",
        description=(
            "Adversaries send targeted emails, messages, or web links containing malicious "
            "URLs designed to lure victims into visiting adversary-controlled sites to steal credentials "
            "or deliver subsequent stage payloads."
        ),
        mitigation_guidance=(
            "Implement enterprise URL filtering, anti-spoofing protocols (DMARC/DKIM/SPF), "
            "multi-factor authentication (MFA/FIDO2), and automated phishing quarantine gateways."
        ),
        detection_methods=(
            "Inspect lexical URL features (credential tokens like login/verify/account, subdomain depth, "
            "and abnormal path lengths) alongside gateway email telemetry."
        ),
        url_reference="https://attack.mitre.org/techniques/T1566/002/",
        related_indicators=[
            "suspicious_keyword_count", "contains_at", "multiple_subdomains",
            "path_length", "query_param_count", "double_slash_count"
        ]
    ),
    "T1566": MitreTechnique(
        technique_id="T1566",
        technique_name="Phishing",
        tactic="Initial Access",
        tactic_id="TA0001",
        description=(
            "Adversaries send broad or untargeted electronic communications containing malicious URLs "
            "to gain initial access, harvest credentials, or execute arbitrary code."
        ),
        mitigation_guidance=(
            "Enforce security awareness training, endpoint web protection, and real-time link rewriting."
        ),
        detection_methods=(
            "Monitor outbound HTTP/HTTPS requests against threat intelligence feeds and lexical classifiers."
        ),
        url_reference="https://attack.mitre.org/techniques/T1566/",
        related_indicators=["suspicious_keyword_count", "url_length", "unique_char_ratio"]
    ),
    "T1204.001": MitreTechnique(
        technique_id="T1204.001",
        technique_name="User Execution: Malicious Link",
        tactic="Execution",
        tactic_id="TA0002",
        description=(
            "An adversary relies on a user clicking a malicious link to trigger execution of code "
            "or direct the user to an exploitation server (drive-by download or browser exploit)."
        ),
        mitigation_guidance=(
            "Use browser isolation (RBI), restrictive Content Security Policies (CSP), and DNS filtering."
        ),
        detection_methods=(
            "Correlate browser click events and proxy logs with newly observed domains and anomalous entropy."
        ),
        url_reference="https://attack.mitre.org/techniques/T1204/001/",
        related_indicators=["url_length", "entropy", "domain_entropy", "is_shortened"]
    ),
    "T1204.002": MitreTechnique(
        technique_id="T1204.002",
        technique_name="User Execution: Malicious File",
        tactic="Execution",
        tactic_id="TA0002",
        description=(
            "An adversary delivers an executable payload, archive, or script via a direct URL link "
            "(e.g., .exe, .zip, .rar, .tar.gz) requiring user action to initiate execution."
        ),
        mitigation_guidance=(
            "Block direct executable and archive downloads at the perimeter gateway; enforce endpoint EDR policies."
        ),
        detection_methods=(
            "Inspect URL extensions, MIME types, and download telemetry for executable signatures."
        ),
        url_reference="https://attack.mitre.org/techniques/T1204/002/",
        related_indicators=["has_exe_or_zip", "path_length", "longest_token_length"]
    ),
    "T1568.002": MitreTechnique(
        technique_id="T1568.002",
        technique_name="Dynamic Resolution: Domain Generation Algorithms (DGA)",
        tactic="Command and Control",
        tactic_id="TA0011",
        description=(
            "Adversaries make use of DGAs to dynamically compute domain names for C2 fallback rendezvous, "
            "evading static domain reputation blocklists."
        ),
        mitigation_guidance=(
            "Deploy recursive DNS firewalls with algorithmic anomaly detection and sinkholing."
        ),
        detection_methods=(
            "Analyze Shannon entropy, consonant-to-vowel ratios, and character n-gram distributions in domain strings."
        ),
        url_reference="https://attack.mitre.org/techniques/T1568/002/",
        related_indicators=[
            "entropy", "domain_entropy", "consonant_ratio", "vowel_ratio",
            "digit_ratio", "longest_token_length"
        ]
    ),
    "T1583.001": MitreTechnique(
        technique_id="T1583.001",
        technique_name="Acquire Infrastructure: Domains",
        tactic="Resource Development",
        tactic_id="TA0042",
        description=(
            "Adversaries purchase or register disposable domain names, often leveraging cheap, high-abuse "
            "top-level domains (TLDs) to stage attacks."
        ),
        mitigation_guidance=(
            "Block or restrict newly registered domains (NRDs < 30 days) and high-risk TLDs at DNS/proxy firewalls."
        ),
        detection_methods=(
            "Monitor WHOIS age, registrar reputation, and suspicious TLD patterns in outbound requests."
        ),
        url_reference="https://attack.mitre.org/techniques/T1583/001/",
        related_indicators=["suspicious_tld", "domain_length", "subdomain_count"]
    ),
    "T1583.008": MitreTechnique(
        technique_id="T1583.008",
        technique_name="Acquire Infrastructure: Malicious URL Shortening Services",
        tactic="Resource Development",
        tactic_id="TA0042",
        description=(
            "Adversaries utilize public or custom URL shortening services (e.g., bit.ly, tinyurl) to obscure "
            "the true destination of malicious endpoints."
        ),
        mitigation_guidance=(
            "Configure enterprise secure web gateways (SWG) to automatically expand and inspect shortened URLs."
        ),
        detection_methods=(
            "Detect known shortening service hostnames and trace multi-hop HTTP 301/302 redirect chains."
        ),
        url_reference="https://attack.mitre.org/techniques/T1583/",
        related_indicators=["is_shortened", "url_length", "domain_length"]
    ),
    "T1090.003": MitreTechnique(
        technique_id="T1090.003",
        technique_name="Proxy: Multi-hop / Direct IP Routing",
        tactic="Command and Control",
        tactic_id="TA0011",
        description=(
            "Adversaries route network traffic directly to raw IP addresses or non-standard ports, "
            "bypassing DNS inspection and domain-level filtering."
        ),
        mitigation_guidance=(
            "Enforce outbound proxy inspection requiring FQDN resolution; deny direct IP connection requests."
        ),
        detection_methods=(
            "Identify raw IPv4/IPv6 host patterns and explicit port bindings in HTTP/HTTPS URLs."
        ),
        url_reference="https://attack.mitre.org/techniques/T1090/003/",
        related_indicators=["contains_ip", "has_port"]
    ),
    "T1036.007": MitreTechnique(
        technique_id="T1036.007",
        technique_name="Masquerading: Double File Extension",
        tactic="Defense Evasion",
        tactic_id="TA0005",
        description=(
            "Adversaries craft URLs with deceptive token sequences, multiple dots, or embedded secondary "
            "extensions to disguise malicious payloads as benign documents."
        ),
        mitigation_guidance=(
            "Enforce deep packet inspection and file-type analysis rather than relying on extension strings."
        ),
        detection_methods=(
            "Count dot occurrences, examine delimiter distributions, and scan for trailing executable extensions."
        ),
        url_reference="https://attack.mitre.org/techniques/T1036/007/",
        related_indicators=["num_dots", "num_hyphens", "longest_token_length", "has_exe_or_zip"]
    ),
    "T1036.008": MitreTechnique(
        technique_id="T1036.008",
        technique_name="Masquerading: Lookalike / Typosquatting Domain",
        tactic="Defense Evasion",
        tactic_id="TA0005",
        description=(
            "Adversaries register lookalike domains with deceptive subdomains or hyphens to mimic trusted brands "
            "and deceive users into trusting malicious URLs."
        ),
        mitigation_guidance=(
            "Register corporate defensive domains and deploy homoglyph-aware DNS inspection."
        ),
        detection_methods=(
            "Check subdomain depth, hyphen counts, Levenshtein distances to trusted roots, and keyword proximity."
        ),
        url_reference="https://attack.mitre.org/techniques/T1036/",
        related_indicators=[
            "subdomain_count", "multiple_subdomains", "num_hyphens",
            "suspicious_keyword_count", "contains_at"
        ]
    ),
    "T1491.001": MitreTechnique(
        technique_id="T1491.001",
        technique_name="Defacement: Internal & External Defacement",
        tactic="Impact",
        tactic_id="TA0040",
        description=(
            "Adversaries modify external or internal web resources to disrupt organizational credibility, "
            "often targeting deep or exposed URL path structures on vulnerable web servers."
        ),
        mitigation_guidance=(
            "Enforce Web Application Firewalls (WAF), file integrity monitoring (FIM), and strict CMS access controls."
        ),
        detection_methods=(
            "Inspect abnormal path depth, token counts, directory traversal sequences, and web server logs."
        ),
        url_reference="https://attack.mitre.org/techniques/T1491/001/",
        related_indicators=["path_length", "path_depth", "num_hyphens", "longest_token_length"]
    ),
}

# Mapping of Threat Categories to Primary MITRE ATT&CK Techniques
THREAT_CATEGORY_MITRE_MAP: Dict[str, List[str]] = {
    "phishing": ["T1566.002", "T1566", "T1204.001", "T1036.008"],
    "malware": ["T1204.002", "T1568.002", "T1090.003", "T1036.007"],
    "defacement": ["T1491.001", "T1583.001"],
    "benign": []
}

# Feature Name to Specific MITRE ATT&CK Technique Mapping
FEATURE_MITRE_MAP: Dict[str, List[str]] = {
    "suspicious_keyword_count": ["T1566.002", "T1036.008"],
    "contains_at": ["T1566.002", "T1036.008"],
    "multiple_subdomains": ["T1036.008", "T1566.002"],
    "subdomain_count": ["T1036.008", "T1566.002"],
    "has_exe_or_zip": ["T1204.002", "T1036.007"],
    "contains_ip": ["T1090.003"],
    "has_port": ["T1090.003"],
    "entropy": ["T1568.002", "T1204.001"],
    "domain_entropy": ["T1568.002"],
    "is_shortened": ["T1583.008", "T1204.001"],
    "suspicious_tld": ["T1583.001"],
    "longest_token_length": ["T1568.002", "T1036.007"],
    "num_hyphens": ["T1036.008", "T1491.001"],
    "num_dots": ["T1036.007", "T1566.002"],
    "path_length": ["T1491.001", "T1566.002"],
    "path_len": ["T1491.001", "T1566.002"],
    "path_depth": ["T1491.001"],
    "url_length": ["T1566", "T1204.001"],
    "url_len": ["T1566", "T1204.001"],
    "abnormal_url": ["T1036.008", "T1566.002"],
    "consonant_ratio": ["T1568.002"],
    "vowel_ratio": ["T1568.002"],
    "digit_ratio": ["T1568.002"],
    "double_slash_count": ["T1566.002", "T1036.008"],
}


class MitreMapper:
    """
    Enterprise-grade MITRE ATT&CK Knowledge Base mapping engine.
    
    Provides thread-safe, deterministic lookup and enrichment of threat categories,
    lexical features, and SHAP attribution vectors with formal MITRE TTPs.
    """

    def __init__(self, catalog: Optional[Dict[str, MitreTechnique]] = None) -> None:
        self._catalog: Dict[str, MitreTechnique] = catalog or MITRE_TECHNIQUE_CATALOG
        logger.info(f"Initialized MitreMapper with {len(self._catalog)} techniques in catalog.")

    def get_technique_by_id(self, technique_id: str) -> Optional[MitreTechnique]:
        """Retrieves a MITRE technique by its identifier (e.g. 'T1566.002')."""
        return self._catalog.get(technique_id.strip())

    def get_techniques_for_threat_category(self, threat_category: str) -> List[MitreTechnique]:
        """Retrieves default MITRE techniques mapped to a threat category."""
        category_clean = threat_category.lower().strip()
        technique_ids = THREAT_CATEGORY_MITRE_MAP.get(category_clean, [])
        return [self._catalog[tid] for tid in technique_ids if tid in self._catalog]

    def map_feature_to_techniques(
        self,
        feature_name: str,
        feature_value: Optional[Any] = None,
        threat_category: Optional[str] = None
    ) -> List[MitreTechnique]:
        """
        Maps an individual lexical feature and its runtime value to associated MITRE ATT&CK techniques.
        
        Args:
            feature_name: Name of the extracted numerical or indicator feature.
            feature_value: Raw runtime value of the feature.
            threat_category: Optional threat category for contextual ranking.
            
        Returns:
            List of matching MitreTechnique objects.
        """
        feat_clean = feature_name.strip()
        matched_ids: List[str] = FEATURE_MITRE_MAP.get(feat_clean, [])

        # Heuristic fallback if feature contains specific keyword stems
        if not matched_ids:
            if "entropy" in feat_clean:
                matched_ids = ["T1568.002"]
            elif "keyword" in feat_clean or "login" in feat_clean or "verify" in feat_clean:
                matched_ids = ["T1566.002"]
            elif "exe" in feat_clean or "zip" in feat_clean or "payload" in feat_clean:
                matched_ids = ["T1204.002"]
            elif "subdomain" in feat_clean:
                matched_ids = ["T1036.008"]
            elif "length" in feat_clean:
                matched_ids = ["T1566"]

        # Prioritize techniques aligned with the threat category while preserving priority order
        if threat_category:
            category_clean = threat_category.lower().strip()
            category_preferred_list = THREAT_CATEGORY_MITRE_MAP.get(category_clean, [])
            category_preferred = set(category_preferred_list)
            # Stable sort: preferred first (by index in category list), then others in original order
            matched_ids = sorted(
                matched_ids,
                key=lambda tid: (
                    0 if tid in category_preferred else 1,
                    category_preferred_list.index(tid) if tid in category_preferred else 999
                )
            )

        return [self._catalog[tid] for tid in matched_ids if tid in self._catalog]

    def enrich_feature_attribution(
        self,
        feature_name: str,
        feature_value: Any,
        shap_value: float,
        threat_category: str
    ) -> Dict[str, Any]:
        """
        Enriches a single SHAP feature attribution tuple with structured MITRE ATT&CK context.
        
        Args:
            feature_name: Name of the feature.
            feature_value: Value of the feature.
            shap_value: Signed SHAP contribution value.
            threat_category: Current predicted threat class name.
            
        Returns:
            Dictionary containing feature attribution, impact direction, and MITRE mapping.
        """
        impact_direction = "POSITIVE" if shap_value > 0 else "NEGATIVE"
        techniques = self.map_feature_to_techniques(feature_name, feature_value, threat_category)

        technique_payloads = [
            {
                "technique_id": t.technique_id,
                "technique_name": t.technique_name,
                "tactic": t.tactic,
                "tactic_id": t.tactic_id,
                "description": t.description,
                "url_reference": t.url_reference
            }
            for t in techniques
        ]

        return {
            "feature_name": feature_name,
            "feature_value": feature_value,
            "shap_value": float(round(shap_value, 6)),
            "impact_direction": impact_direction,
            "mitre_techniques": technique_payloads,
            "primary_mitre_technique": technique_payloads[0] if technique_payloads else None
        }

    def summarize_mitre_coverage(self, technique_ids: List[str]) -> List[Dict[str, Any]]:
        """Summarizes unique techniques and their corresponding tactics for reporting."""
        seen: Set[str] = set()
        summary: List[Dict[str, Any]] = []
        for tid in technique_ids:
            if tid in self._catalog and tid not in seen:
                seen.add(tid)
                t = self._catalog[tid]
                summary.append({
                    "technique_id": t.technique_id,
                    "technique_name": t.technique_name,
                    "tactic": t.tactic,
                    "tactic_id": t.tactic_id,
                    "url_reference": t.url_reference
                })
        return summary


# Module-level cached instance
_GLOBAL_MITRE_MAPPER: Optional[MitreMapper] = None


def get_mitre_mapper() -> MitreMapper:
    """Provides a thread-safe singleton instance of the MitreMapper."""
    global _GLOBAL_MITRE_MAPPER
    if _GLOBAL_MITRE_MAPPER is None:
        _GLOBAL_MITRE_MAPPER = MitreMapper()
    return _GLOBAL_MITRE_MAPPER
