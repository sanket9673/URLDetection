"""
Explainability and MITRE ATT&CK Mapping Module for URL Threat Detection.
"""

from src.explainability.mitre_mapper import (
    MitreMapper,
    MitreTechnique,
    get_mitre_mapper,
    MITRE_TECHNIQUE_CATALOG,
    THREAT_CATEGORY_MITRE_MAP,
    FEATURE_MITRE_MAP
)
from src.explainability.shap_engine import (
    ShapEngine,
    get_shap_engine,
    CLASS_NAMES,
    CLASS_INDICES
)
from src.explainability.graph_rag import (
    GraphRAGEngine,
    GraphRAGContext,
    StructuralIntelligence,
    SubgraphNode,
    SubgraphEdge,
    get_graph_rag_engine
)

__all__ = [
    "MitreMapper",
    "MitreTechnique",
    "get_mitre_mapper",
    "MITRE_TECHNIQUE_CATALOG",
    "THREAT_CATEGORY_MITRE_MAP",
    "FEATURE_MITRE_MAP",
    "ShapEngine",
    "get_shap_engine",
    "CLASS_NAMES",
    "CLASS_INDICES",
    "GraphRAGEngine",
    "GraphRAGContext",
    "StructuralIntelligence",
    "SubgraphNode",
    "SubgraphEdge",
    "get_graph_rag_engine"
]
