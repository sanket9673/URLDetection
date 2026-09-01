import os
import json
import pickle
import time
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from src.logger_config import get_logger

logger = get_logger(__name__)

def train_lightgbm(
    data_path="data/processed/feature_dataset.parquet",
    model_output_path="models/lightgbm_model.pkl",
    metrics_output_path="outputs/lightgbm_metrics.json"
):
    start_time = time.time()
    logger.info("Starting LightGBM Training Pipeline...")
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Processed feature dataset not found at {data_path}")
        
    logger.info(f"Loading feature dataset from {data_path}")
    df = pd.read_parquet(data_path)
    
    target_col = 'target' if 'target' in df.columns else 'label'
    if target_col not in df.columns:
        raise ValueError("Target column not found in dataset.")
        
    exclude_cols = ['url', 'type', 'target', 'label', 'registered_domain', 'tld']
    numeric_df = df.select_dtypes(include=['number', 'bool'])
    feature_cols = [c for c in numeric_df.columns if c not in exclude_cols]
    
    logger.info(f"Extracted {len(feature_cols)} feature columns for training.")
    
    X = df[feature_cols]
    y = df[target_col]
    
    logger.info("Executing strict stratified 70% Train, 15% Validation, and 15% Test split...")
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=42
    )
    
    logger.info(f"Train shape: {X_train.shape}, Val shape: {X_val.shape}, Test shape: {X_test.shape}")
    
    logger.info("Initializing LGBMClassifier with multiclass objective...")
    clf = LGBMClassifier(
        objective='multiclass',
        num_class=4,
        n_estimators=150,
        learning_rate=0.05,
        num_leaves=31,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    
    logger.info("Fitting LightGBM model on training dataset...")
    clf.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[]
    )
    
    logger.info("Evaluating model on Test split...")
    y_pred = clf.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average='macro')
    weighted_f1 = f1_score(y_test, y_pred, average='weighted')
    per_class_f1 = f1_score(y_test, y_pred, average=None).tolist()
    per_class_prec = precision_score(y_test, y_pred, average=None).tolist()
    per_class_rec = recall_score(y_test, y_pred, average=None).tolist()
    
    metrics = {
        "accuracy": float(acc),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "per_class_f1": [float(x) for x in per_class_f1],
        "per_class_precision": [float(x) for x in per_class_prec],
        "per_class_recall": [float(x) for x in per_class_rec]
    }
    
    logger.info(f"Test Accuracy: {acc:.4f}, Test Macro F1: {macro_f1:.4f}, Test Weighted F1: {weighted_f1:.4f}")
    logger.info(f"Per-class F1: {[round(x, 4) for x in per_class_f1]}")
    
    # Save model weights
    os.makedirs(os.path.dirname(model_output_path), exist_ok=True)
    with open(model_output_path, 'wb') as f:
        pickle.dump(clf, f)
    logger.info(f"Saved trained LightGBM model weights to {model_output_path}")
    
    # Save metrics JSON
    os.makedirs(os.path.dirname(metrics_output_path), exist_ok=True)
    with open(metrics_output_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    logger.info(f"Saved evaluation metrics JSON to {metrics_output_path}")
    
    time_taken = time.time() - start_time
    logger.info(f"LightGBM Training Completed in {time_taken:.2f} seconds.")
    return clf, metrics

if __name__ == "__main__":
    train_lightgbm()
