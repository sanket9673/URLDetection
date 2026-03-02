import os
import json
import logging
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.model_selection import train_test_split

try:
    from src.logger_config import get_logger
    logger = get_logger(__name__)
except ImportError:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

def hybrid_fusion(
    data_path="data/processed/graph_features.parquet",
    model_path="models/lightgbm_model.pkl",
    output_path="outputs/hybrid_metrics.json"
):
    logger.info("Starting Hybrid Fusion Process...")
    
    # Handle missing files safely
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data not found: {data_path}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")

    # Load data
    logger.info(f"Loading dataset from {data_path}")
    df = pd.read_parquet(data_path)
    
    # Load model
    logger.info(f"Loading LightGBM model from {model_path}")
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
        
    target_col = 'target' if 'target' in df.columns else 'label'
    if target_col not in df.columns:
        raise ValueError("Target column not found.")

    classes = model.classes_
    num_classes = len(classes)
    logger.info(f"Detected {num_classes} classes from LightGBM model: {classes}")

    logger.info("Recreating Train/Val/Test splits to match training pipeline...")
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # Stratified split: 70% Train, 30% Temp
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=42
    )
    # 50% Val, 50% Test of Temp (15% each total)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=42
    )

    feature_names = model.feature_name_
    logger.info("Generating LightGBM Probabilities (P_feature_i)...")
    
    try:
        # We only pass features that LightGBM expects
        P_feature_val = model.predict_proba(X_val[feature_names])
        P_feature_test = model.predict_proba(X_test[feature_names])
    except Exception as e:
        logger.error(f"Failed to generate LightGBM predictions: {e}")
        raise

    logger.info("Extracting Graph Probabilities (P_graph_i) and Normalizing...")
    # domain_graph.py guarantees 'domain_class_{c}_prob' exists for each class present in training
    graph_cols = [f'domain_class_{c}_prob' for c in sorted(y.unique())]
    if not all(col in X_val.columns for col in graph_cols):
        # Fallback to R_graph_class_
        graph_cols = [f'R_graph_class_{c}' for c in sorted(y.unique())]
        if not all(col in X_val.columns for col in graph_cols):
            raise KeyError(f"Graph probability columns not found in dataset. Expected {graph_cols}")
            
    P_graph_val = X_val[graph_cols].values
    P_graph_test = X_test[graph_cols].values
    
    # 1. Normalize graph probabilities
    # Add epsilon to prevent divide by zero
    P_graph_val_sum = np.sum(P_graph_val, axis=1, keepdims=True)
    P_graph_val = P_graph_val / (P_graph_val_sum + 1e-9)
    
    P_graph_test_sum = np.sum(P_graph_test, axis=1, keepdims=True)
    P_graph_test = P_graph_test / (P_graph_test_sum + 1e-9)

    # 2. Tune alpha in [0.3, 0.5, 0.7]
    alphas = [0.3, 0.5, 0.7]
    best_alpha = None
    best_val_f1 = -1
    tuning_log = []

    logger.info("Tuning alpha on validation set...")
    for alpha in alphas:
        beta = 1.0 - alpha
        
        P_final_val = alpha * P_feature_val + beta * P_graph_val
        y_val_pred_indices = np.argmax(P_final_val, axis=1)
        
        # Map indices back to actual class labels
        y_val_pred = classes[y_val_pred_indices]
        
        val_f1 = f1_score(y_val, y_val_pred, average='macro')
        
        log_entry = {
            "alpha": float(alpha),
            "validation_macro_f1": float(val_f1)
        }
        tuning_log.append(log_entry)
        
        logger.info(f"Alpha: {alpha:.1f} (Beta: {beta:.1f}) -> Validation Macro F1: {val_f1:.4f}")
        
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_alpha = alpha

    # 3. Choose best based on Macro F1 on validation set
    logger.info(f"Best Alpha selected: {best_alpha:.1f} with Val Macro F1: {best_val_f1:.4f}")

    # 4. Evaluate on test set
    logger.info("Evaluating optimal fusion on Test set...")
    best_beta = 1.0 - best_alpha
    P_final_test = best_alpha * P_feature_test + best_beta * P_graph_test
    
    y_test_pred_indices = np.argmax(P_final_test, axis=1)
    y_test_pred = classes[y_test_pred_indices]
    
    test_f1 = f1_score(y_test, y_test_pred, average='macro')
    test_cm = confusion_matrix(y_test, y_test_pred).tolist()
    
    logger.info(f"Final Test Macro F1: {test_f1:.4f}")
    
    metrics = {
        "alpha_tested": tuning_log,
        "best_alpha": float(best_alpha),
        "validation_best_f1": float(best_val_f1),
        "test_f1": float(test_f1),
        "test_confusion_matrix": test_cm
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(metrics, f, indent=4)
        
    logger.info(f"Successfully saved test metrics to {output_path}")
    logger.info("Hybrid Fusion Process Completed.")
    return metrics

if __name__ == "__main__":
    try:
        hybrid_fusion()
    except Exception as e:
        logger.error(f"Hybrid fusion failed: {str(e)}", exc_info=True)
        raise
