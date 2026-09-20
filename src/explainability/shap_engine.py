"""
SHAP Attribution Engine for URL Threat Classification.

Computes exact local Shapley feature contributions from the pre-trained LightGBM model
and enriches lexical attributions with the MITRE ATT&CK Knowledge Base.
"""

from __future__ import annotations

import json
import logging
import os
import pickle
import threading
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from src.explainability.mitre_mapper import MitreMapper, get_mitre_mapper
from src.feature_engineering.feature_builder import FeatureBuilder

try:
    from src.logger_config import get_logger
    logger = get_logger(__name__)
except ImportError:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    logger = logging.getLogger(__name__)


# Standard Threat Classes
CLASS_NAMES: Dict[int, str] = {
    0: "benign",
    1: "defacement",
    2: "phishing",
    3: "malware"
}
CLASS_INDICES: Dict[str, int] = {v: k for k, v in CLASS_NAMES.items()}


class ShapEngine:
    """
    High-throughput, enterprise-grade SHAP Attribution Engine for URL classification.
    
    Provides thread-safe local feature attributions using LightGBM's exact C++ TreeSHAP
    implementation, integrated with MITRE ATT&CK tactical mapping.
    """

    def __init__(
        self,
        model_path: str = "models/lightgbm_model.pkl",
        mitre_mapper: Optional[MitreMapper] = None
    ) -> None:
        """
        Initializes the SHAP Attribution Engine.
        
        Args:
            model_path: Path to the serialized LightGBM model binary.
            mitre_mapper: Optional custom MitreMapper instance.
        """
        self.model_path = model_path
        self._lock = threading.Lock()
        self._mitre_mapper = mitre_mapper or get_mitre_mapper()
        
        self._model = None
        self._feature_names: List[str] = []
        self._num_features: int = 0
        self._num_classes: int = 4
        self._classes: np.ndarray = np.array([0, 1, 2, 3])
        
        self._load_model()
        # Feature builder helper instance
        self._feature_builder = FeatureBuilder(raw_data_path="", output_path="")

    def _load_model(self) -> None:
        """Loads and caches the LightGBM model binary with thread safety."""
        with self._lock:
            if self._model is not None:
                return

            if not os.path.exists(self.model_path):
                raise FileNotFoundError(
                    f"Trained LightGBM model binary not found at '{self.model_path}'. "
                    "Ensure models/lightgbm_model.pkl exists or run model training."
                )

            logger.info(f"Loading LightGBM model from {self.model_path} into ShapEngine...")
            with open(self.model_path, "rb") as f:
                self._model = pickle.load(f)

            if hasattr(self._model, "feature_name_") and self._model.feature_name_ is not None:
                self._feature_names = list(self._model.feature_name_)
            else:
                raise AttributeError("Loaded model does not contain valid 'feature_name_' attributes.")

            if hasattr(self._model, "classes_") and self._model.classes_ is not None:
                self._classes = np.array(self._model.classes_)
                self._num_classes = len(self._classes)

            if hasattr(self._model, "set_params"):
                try:
                    self._model.set_params(n_jobs=1)
                except Exception:
                    pass

            self._num_features = len(self._feature_names)
            logger.info(
                f"ShapEngine successfully initialized: {self._num_features} features, "
                f"{self._num_classes} classes."
            )

    @property
    def feature_names(self) -> List[str]:
        """Returns the expected list of feature names."""
        return self._feature_names.copy()

    def extract_features_from_url(self, url: str) -> pd.DataFrame:
        """
        Extracts and aligns numerical lexical features for a raw URL string.
        
        Args:
            url: The input URL string to inspect.
            
        Returns:
            DataFrame with a single row aligned to model.feature_name_.
        """
        url_clean = str(url).strip()
        if not url_clean:
            url_clean = "about:blank"

        raw_df = pd.DataFrame({"url": [url_clean], "type": ["benign"]})
        cleaned_df = self._feature_builder.validate_and_clean(raw_df)
        featured_df = self._feature_builder.build_features(cleaned_df)
        
        # Select numeric columns and reindex exactly to expected model features
        numeric_df = featured_df.select_dtypes(include=["number", "bool"])
        aligned_df = numeric_df.reindex(columns=self._feature_names, fill_value=0.0)
        return aligned_df

    def extract_features_from_urls(self, urls: List[str]) -> pd.DataFrame:
        """
        Extracts and aligns numerical lexical features for a batch of URL strings.
        
        Args:
            urls: List of URL strings.
            
        Returns:
            DataFrame with rows aligned to model.feature_name_.
        """
        if not urls:
            return pd.DataFrame(columns=self._feature_names)

        urls_clean = [str(u).strip() if str(u).strip() else "about:blank" for u in urls]
        raw_df = pd.DataFrame({"url": urls_clean, "type": ["benign"] * len(urls_clean)})
        cleaned_df = self._feature_builder.validate_and_clean(raw_df)
        featured_df = self._feature_builder.build_features(cleaned_df)
        
        numeric_df = featured_df.select_dtypes(include=["number", "bool"])
        aligned_df = numeric_df.reindex(columns=self._feature_names, fill_value=0.0)
        return aligned_df

    def _compute_raw_shap(self, feature_matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Computes exact TreeSHAP values using LightGBM's native C++ implementation.
        
        Args:
            feature_matrix: 2D numpy array of shape (n_samples, n_features).
            
        Returns:
            Tuple of:
              - shap_values: 3D array of shape (n_samples, n_classes, n_features)
              - base_values: 2D array of shape (n_samples, n_classes) representing expected values (bias).
        """
        n_samples = feature_matrix.shape[0]
        # LightGBM booster_.predict with pred_contrib=True returns (n_samples, n_classes * (n_features + 1))
        raw_contrib = self._model.booster_.predict(feature_matrix, pred_contrib=True)
        
        # Reshape to (n_samples, n_classes, n_features + 1)
        reshaped = raw_contrib.reshape(n_samples, self._num_classes, self._num_features + 1)
        
        # First n_features are Shapley values; the final column is base value (bias)
        shap_values = reshaped[:, :, :self._num_features]
        base_values = reshaped[:, :, self._num_features]
        
        return shap_values, base_values

    def explain_url(
        self,
        url: str,
        target_class: Optional[Union[int, str]] = None,
        top_k: int = 5
    ) -> Dict[str, Any]:
        """
        Generates structured local SHAP attribution and MITRE ATT&CK mapping for a single URL.
        
        Args:
            url: The URL string to inspect.
            target_class: Optional specific class index or name (default: model's predicted class).
            top_k: Number of top contributing features to return (default: 5).
            
        Returns:
            Structured JSON-compatible dictionary (Top-5 Lexical SHAP JSON).
        """
        start_time = time.perf_counter()
        
        # 1. Feature Extraction
        df_features = self.extract_features_from_url(url)
        feat_matrix = df_features.values.astype(np.float64)
        
        # 2. Fast & Thread-Safe Probability and Class Prediction via Booster
        probabilities = self._model.booster_.predict(feat_matrix, num_iteration=getattr(self._model, "best_iteration_", None))[0]
        pred_class_idx = int(np.argmax(probabilities))
        pred_class_name = CLASS_NAMES.get(pred_class_idx, str(pred_class_idx))
        
        # Resolve target class for attribution
        if target_class is not None:
            if isinstance(target_class, str):
                class_to_explain = CLASS_INDICES.get(target_class.lower().strip(), pred_class_idx)
            else:
                class_to_explain = int(target_class)
        else:
            class_to_explain = pred_class_idx

        target_class_name = CLASS_NAMES.get(class_to_explain, str(class_to_explain))
        
        # 3. Exact TreeSHAP Computation
        shap_values_3d, base_values_2d = self._compute_raw_shap(feat_matrix)
        
        # Extract 1D SHAP values for the target class: shape (n_features,)
        sample_shap = shap_values_3d[0, class_to_explain, :]
        sample_base = float(base_values_2d[0, class_to_explain])
        raw_feature_values = df_features.iloc[0].to_dict()
        
        # 4. Rank Features by Absolute Contribution Magnitude
        abs_shap = np.abs(sample_shap)
        top_indices = np.argsort(abs_shap)[::-1][:top_k]
        
        top_features: List[Dict[str, Any]] = []
        mitre_technique_ids: List[str] = []
        
        for rank, idx in enumerate(top_indices, start=1):
            feat_name = self._feature_names[idx]
            feat_val = raw_feature_values[feat_name]
            shap_val = float(sample_shap[idx])
            
            # Format feature value for clean JSON output
            if isinstance(feat_val, (np.floating, float)):
                feat_val_formatted = round(float(feat_val), 4)
            elif isinstance(feat_val, (np.integer, int)):
                feat_val_formatted = int(feat_val)
            elif isinstance(feat_val, (np.bool_, bool)):
                feat_val_formatted = bool(feat_val)
            else:
                feat_val_formatted = feat_val
                
            enriched = self._mitre_mapper.enrich_feature_attribution(
                feature_name=feat_name,
                feature_value=feat_val_formatted,
                shap_value=shap_val,
                threat_category=target_class_name
            )
            
            # Collect technique IDs for high-level tactical summary
            for tech in enriched.get("mitre_techniques", []):
                mitre_technique_ids.append(tech["technique_id"])
                
            top_features.append({
                "rank": rank,
                "feature_name": feat_name,
                "feature_value": feat_val_formatted,
                "shap_value": enriched["shap_value"],
                "impact_direction": enriched["impact_direction"],
                "mitre_techniques": enriched["mitre_techniques"],
                "primary_mitre_technique": enriched["primary_mitre_technique"]
            })
            
        # 5. MITRE Coverage Summary
        mitre_summary = self._mitre_mapper.summarize_mitre_coverage(mitre_technique_ids)
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        
        # Construct final JSON payload
        response: Dict[str, Any] = {
            "url": url,
            "prediction": {
                "predicted_class": pred_class_idx,
                "predicted_class_name": pred_class_name,
                "confidence_percent": round(float(probabilities[pred_class_idx]) * 100.0, 2),
                "all_class_probabilities": {
                    CLASS_NAMES.get(i, str(i)): round(float(prob), 4)
                    for i, prob in enumerate(probabilities)
                }
            },
            "attribution_target": {
                "class_index": class_to_explain,
                "class_name": target_class_name,
                "base_value": round(sample_base, 6),
                "total_shap_sum": round(float(np.sum(sample_shap)), 6)
            },
            "top_contributing_features": top_features,
            "mitre_attack_summary": mitre_summary,
            "metadata": {
                "engine": "LightGBM TreeSHAP C++",
                "features_analyzed": self._num_features,
                "top_k": top_k,
                "computation_time_ms": round(elapsed_ms, 2)
            }
        }
        
        return response

    def explain_batch_urls(
        self,
        urls: List[str],
        target_class: Optional[Union[int, str]] = None,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Generates batch SHAP explanations for high-throughput enterprise ingestion.
        
        Args:
            urls: List of URL strings to inspect.
            target_class: Optional target class override.
            top_k: Number of top contributing features per URL.
            
        Returns:
            List of structured explanation payloads.
        """
        if not urls:
            return []

        # Vectorized feature extraction for the entire batch
        df_batch = self.extract_features_from_urls(urls)
        feat_matrix = df_batch.values.astype(np.float64)
        
        # Fast & Thread-Safe Batch inference
        batch_probs = self._model.booster_.predict(feat_matrix, num_iteration=getattr(self._model, "best_iteration_", None))
        shap_values_3d, base_values_2d = self._compute_raw_shap(feat_matrix)
        
        results: List[Dict[str, Any]] = []
        
        for i, url in enumerate(urls):
            probabilities = batch_probs[i]
            pred_class_idx = int(np.argmax(probabilities))
            pred_class_name = CLASS_NAMES.get(pred_class_idx, str(pred_class_idx))
            
            if target_class is not None:
                if isinstance(target_class, str):
                    class_to_explain = CLASS_INDICES.get(target_class.lower().strip(), pred_class_idx)
                else:
                    class_to_explain = int(target_class)
            else:
                class_to_explain = pred_class_idx
                
            target_class_name = CLASS_NAMES.get(class_to_explain, str(class_to_explain))
            sample_shap = shap_values_3d[i, class_to_explain, :]
            sample_base = float(base_values_2d[i, class_to_explain])
            raw_feature_values = df_batch.iloc[i].to_dict()
            
            abs_shap = np.abs(sample_shap)
            top_indices = np.argsort(abs_shap)[::-1][:top_k]
            
            top_features: List[Dict[str, Any]] = []
            mitre_technique_ids: List[str] = []
            
            for rank, idx in enumerate(top_indices, start=1):
                feat_name = self._feature_names[idx]
                feat_val = raw_feature_values[feat_name]
                shap_val = float(sample_shap[idx])
                
                if isinstance(feat_val, (np.floating, float)):
                    feat_val_formatted = round(float(feat_val), 4)
                elif isinstance(feat_val, (np.integer, int)):
                    feat_val_formatted = int(feat_val)
                elif isinstance(feat_val, (np.bool_, bool)):
                    feat_val_formatted = bool(feat_val)
                else:
                    feat_val_formatted = feat_val
                    
                enriched = self._mitre_mapper.enrich_feature_attribution(
                    feature_name=feat_name,
                    feature_value=feat_val_formatted,
                    shap_value=shap_val,
                    threat_category=target_class_name
                )
                for tech in enriched.get("mitre_techniques", []):
                    mitre_technique_ids.append(tech["technique_id"])
                    
                top_features.append({
                    "rank": rank,
                    "feature_name": feat_name,
                    "feature_value": feat_val_formatted,
                    "shap_value": enriched["shap_value"],
                    "impact_direction": enriched["impact_direction"],
                    "mitre_techniques": enriched["mitre_techniques"],
                    "primary_mitre_technique": enriched["primary_mitre_technique"]
                })
                
            mitre_summary = self._mitre_mapper.summarize_mitre_coverage(mitre_technique_ids)
            
            results.append({
                "url": url,
                "prediction": {
                    "predicted_class": pred_class_idx,
                    "predicted_class_name": pred_class_name,
                    "confidence_percent": round(float(probabilities[pred_class_idx]) * 100.0, 2),
                    "all_class_probabilities": {
                        CLASS_NAMES.get(j, str(j)): round(float(prob), 4)
                        for j, prob in enumerate(probabilities)
                    }
                },
                "attribution_target": {
                    "class_index": class_to_explain,
                    "class_name": target_class_name,
                    "base_value": round(sample_base, 6),
                    "total_shap_sum": round(float(np.sum(sample_shap)), 6)
                },
                "top_contributing_features": top_features,
                "mitre_attack_summary": mitre_summary
            })
            
        return results


# Module-level cached instance
_GLOBAL_SHAP_ENGINE: Optional[ShapEngine] = None
_ENGINE_LOCK = threading.Lock()


def get_shap_engine(model_path: str = "models/lightgbm_model.pkl") -> ShapEngine:
    """
    Returns a thread-safe singleton instance of the ShapEngine.
    
    Args:
        model_path: Path to the LightGBM model binary.
        
    Returns:
        Cached ShapEngine instance.
    """
    global _GLOBAL_SHAP_ENGINE
    if _GLOBAL_SHAP_ENGINE is None:
        with _ENGINE_LOCK:
            if _GLOBAL_SHAP_ENGINE is None:
                _GLOBAL_SHAP_ENGINE = ShapEngine(model_path=model_path)
    return _GLOBAL_SHAP_ENGINE
