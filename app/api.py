import os
import sys
import time
import json
import pickle
import logging
import tldextract
import numpy as np
import pandas as pd
import lightgbm
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Dict

# Centralized logging configuration
try:
    from src.logger_config import get_logger
    logger = get_logger(__name__)
except ImportError:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

# Make sure project root is in sys.path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.feature_engineering.feature_builder import FeatureBuilder
from src.graph.gnn_train import HeteroGraphSAGE, predict_gnn_dynamic

app = FastAPI(title="Hybrid URL Intelligence API", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global models state
models = {
    "lightgbm": None,
    "gnn_model": None,
    "gnn_data": None,
    "gnn_mappings": None,
    "alpha": 0.7
}

CLASSES = ['benign', 'defacement', 'phishing', 'malware']
CLASS_MAP = {
    'benign': 'Benign',
    'defacement': 'Defacement',
    'phishing': 'Phishing',
    'malware': 'Malware'
}

def load_models():
    logger.info("Loading model artifacts...")
    
    # Paths
    lgb_path = os.path.join(project_root, "models", "lightgbm_model.pkl")
    gnn_data_path = os.path.join(project_root, "models", "gnn_graph_data.pt")
    gnn_mappings_path = os.path.join(project_root, "models", "gnn_mappings.pkl")
    gnn_model_path = os.path.join(project_root, "models", "graphsage_model.pth")
    metrics_path = os.path.join(project_root, "outputs", "hybrid_metrics.json")
    
    try:
        # Load LightGBM
        if os.path.exists(lgb_path):
            with open(lgb_path, "rb") as f:
                models["lightgbm"] = pickle.load(f)
            logger.info("LightGBM model loaded successfully.")
        else:
            logger.error(f"LightGBM model not found at {lgb_path}")
            
        # Load GNN structures
        if os.path.exists(gnn_data_path) and os.path.exists(gnn_mappings_path) and os.path.exists(gnn_model_path):
            models["gnn_data"] = torch.load(gnn_data_path, weights_only=False)
            with open(gnn_mappings_path, "rb") as f:
                models["gnn_mappings"] = pickle.load(f)
                
            device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
            gnn_model = HeteroGraphSAGE(hidden_channels=64, out_channels=4, metadata=models["gnn_data"].metadata())
            gnn_model.load_state_dict(torch.load(gnn_model_path, map_location=device, weights_only=True))
            gnn_model.to(device)
            gnn_model.eval()
            
            models["gnn_model"] = gnn_model
            models["gnn_data"] = models["gnn_data"].to(device)
            logger.info("GNN model and graph data loaded successfully.")
        else:
            logger.error("GNN artifacts missing in models/ folder.")
            
        # Load alpha
        if os.path.exists(metrics_path):
            with open(metrics_path, "r") as f:
                metrics_data = json.load(f)
                models["alpha"] = metrics_data.get("best_alpha", 0.7)
                logger.info(f"Loaded tuned alpha: {models['alpha']}")
                
    except Exception as e:
        logger.error(f"Failed to load models during startup: {str(e)}", exc_info=True)

@app.on_event("startup")
def startup_event():
    load_models()

# Request Models
class URLQuery(BaseModel):
    url: str = Field(..., description="The URL to analyze", example="https://google.com")

class BatchURLQuery(BaseModel):
    urls: List[str] = Field(..., description="List of URLs to analyze", example=["https://google.com", "http://phishing-site.xyz"])

# Response Schemas
class URLPredictionResponse(BaseModel):
    predicted_class: str = Field(..., description="Predicted URL category (Benign, Phishing, Defacement, Malware)")
    probabilities: Dict[str, float] = Field(..., description="Predicted probabilities for each class")
    confidence_score: float = Field(..., description="Confidence score of the prediction (percentage)")
    processing_latency_ms: float = Field(..., description="Latency in milliseconds")

class BatchURLPredictionResponse(BaseModel):
    predictions: List[URLPredictionResponse] = Field(..., description="List of URL prediction responses")

# Global error handling
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error(f"Unhandled exception during request: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": f"Internal Server Error: {str(exc)}"}
    )

def predict_single(url: str) -> URLPredictionResponse:
    t_start = time.time()
    
    if not models["lightgbm"] or not models["gnn_model"]:
        raise HTTPException(status_code=503, detail="Model files are offline or not loaded.")
        
    try:
        # Standardize URL path & prefix (Mitigate protocol bias)
        import re
        realignment_url = re.sub(r'(?i)^https?://(www\.)?', '', url)
        realignment_url = realignment_url.rstrip('/')
        
        # Build features
        df_input = pd.DataFrame([{'url': realignment_url, 'type': 'unknown'}])
        builder = FeatureBuilder(raw_data_path="", output_path="")
        df_clean = builder.validate_and_clean(df_input)
        
        if df_clean.empty:
            raise HTTPException(status_code=400, detail="Invalid URL format. Parsing failed.")
            
        df_features = builder.build_features(df_clean)
        model_features = df_features[models["lightgbm"].feature_name_]
        
        # LightGBM inference
        P_feature = models["lightgbm"].predict_proba(model_features)[0]
        
        # GNN Inference with dynamic injection & cold-start fallback
        P_graph = predict_gnn_dynamic(
            [url], 
            df_features, 
            models["gnn_model"], 
            models["gnn_data"], 
            models["gnn_mappings"]
        )[0]
        
        # Hybrid Fusion
        alpha = models["alpha"]
        beta = 1.0 - alpha
        P_final = alpha * P_feature + beta * P_graph
        P_final = P_final / np.sum(P_final)
        
        pred_idx = np.argmax(P_final)
        pred_label = CLASSES[pred_idx]
        confidence = float(P_final[pred_idx] * 100)
        
        probabilities_dict = {CLASS_MAP[c]: float(P_final[i]) for i, c in enumerate(CLASSES)}
        
        latency = (time.time() - t_start) * 1000
        
        return URLPredictionResponse(
            predicted_class=CLASS_MAP[pred_label],
            probabilities=probabilities_dict,
            confidence_score=confidence,
            processing_latency_ms=latency
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction failed for URL {url}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=400, detail=f"Failed to process URL: {str(e)}")

def predict_batch(urls: List[str]) -> List[URLPredictionResponse]:
    t_start = time.time()
    
    if not models["lightgbm"] or not models["gnn_model"]:
        raise HTTPException(status_code=503, detail="Model files are offline or not loaded.")
        
    try:
        import re
        realignment_urls = []
        for url in urls:
            r_url = re.sub(r'(?i)^https?://(www\.)?', '', url)
            r_url = r_url.rstrip('/')
            realignment_urls.append(r_url)
            
        # Build features for all URLs in batch using vectorized FeatureBuilder
        df_inputs = pd.DataFrame([{'url': u, 'type': 'unknown'} for u in realignment_urls])
        builder = FeatureBuilder(raw_data_path="", output_path="")
        df_clean = builder.validate_and_clean(df_inputs)
        
        if df_clean.empty:
            raise HTTPException(status_code=400, detail="All URLs in batch are malformed or invalid.")
            
        df_features = builder.build_features(df_clean)
        model_features = df_features[models["lightgbm"].feature_name_]
        
        # LightGBM predictions for batch
        P_features = models["lightgbm"].predict_proba(model_features)
        
        # Vectorized GNN predictions for batch
        P_graphs = predict_gnn_dynamic(
            urls, 
            df_features, 
            models["gnn_model"], 
            models["gnn_data"], 
            models["gnn_mappings"]
        )
        
        # Hybrid Fusion for batch
        alpha = models["alpha"]
        beta = 1.0 - alpha
        
        P_finals = alpha * P_features + beta * P_graphs
        P_finals = P_finals / np.sum(P_finals, axis=1, keepdims=True)
        
        responses = []
        for i, url in enumerate(urls):
            p_final = P_finals[i]
            pred_idx = np.argmax(p_final)
            pred_label = CLASSES[pred_idx]
            confidence = float(p_final[pred_idx] * 100)
            probabilities_dict = {CLASS_MAP[c]: float(p_final[j]) for j, c in enumerate(CLASSES)}
            
            responses.append(URLPredictionResponse(
                predicted_class=CLASS_MAP[pred_label],
                probabilities=probabilities_dict,
                confidence_score=confidence,
                processing_latency_ms=(time.time() - t_start) * 1000
            ))
            
        return responses
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Batch prediction failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=400, detail=f"Batch processing failed: {str(e)}")

# Endpoints
@app.get("/health")
def health():
    return {
        "status": "healthy",
        "models_loaded": {
            "lightgbm": models["lightgbm"] is not None,
            "gnn": models["gnn_model"] is not None
        }
    }

@app.post("/predict", response_model=URLPredictionResponse)
def predict(query: URLQuery):
    return predict_single(query.url)

@app.post("/batch-predict", response_model=BatchURLPredictionResponse)
def batch_predict(query: BatchURLQuery):
    if not query.urls:
        raise HTTPException(status_code=400, detail="URL list cannot be empty.")
    preds = predict_batch(query.urls)
    return BatchURLPredictionResponse(predictions=preds)
