import os
import sys
import json
import subprocess
import argparse
from src.logger_config import get_logger

logger = get_logger(__name__)

def run_step(step_name: str, script_path: str, base_dir: str, args: list = None):
    logger.info(f"--- Starting Step: {step_name} ---")
    try:
        # Run the scripts as a subprocess
        env = os.environ.copy()
        env["PYTHONPATH"] = base_dir
        cmd = [sys.executable, script_path]
        if args:
            cmd.extend(args)
        result = subprocess.run(cmd, check=True, capture_output=True, text=True, cwd=base_dir, env=env)
        # Log the output
        for line in result.stdout.splitlines():
            logger.info(f"[{step_name} STDOUT] {line}")
        logger.info(f"--- Completed Step: {step_name} ---")
    except subprocess.CalledProcessError as e:
        logger.error(f"Error in {step_name} at {script_path}: {e}")
        logger.error(f"[{step_name} STDERR]\n{e.stderr}")
        logger.error(f"[{step_name} STDOUT]\n{e.stdout}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error in {step_name} at {script_path}: {e}")
        sys.exit(1)

def print_final_results(base_dir: str):
    logger.info("Printing final results...")
    try:
        lightgbm_metrics_path = os.path.join(base_dir, "outputs", "lightgbm_metrics.json")
        hybrid_metrics_path = os.path.join(base_dir, "outputs", "hybrid_metrics.json")
        
        with open(lightgbm_metrics_path, 'r') as f:
            lgb_metrics = json.load(f)
            
        with open(hybrid_metrics_path, 'r') as f:
            hybrid_metrics = json.load(f)
            
        lgb_macro_f1 = lgb_metrics.get("macro_f1", "N/A")
        if isinstance(lgb_macro_f1, float):
            lgb_macro_f1 = f"{lgb_macro_f1:.4f}"
            
        hybrid_macro_f1 = hybrid_metrics.get("test_f1", "N/A")
        if isinstance(hybrid_macro_f1, float):
            hybrid_macro_f1 = f"{hybrid_macro_f1:.4f}"
            
        per_class_f1 = lgb_metrics.get("per_class_f1", "N/A")
        if isinstance(per_class_f1, list):
            per_class_f1 = [round(x, 4) for x in per_class_f1]
        
        print("\n================================")
        print("FINAL RESULTS")
        print("================================")
        print(f"LightGBM Macro F1: {lgb_macro_f1}")
        print(f"Hybrid Macro F1: {hybrid_macro_f1}")
        print(f"Per-class scores: {per_class_f1}")
        print("================================\n")
        
    except FileNotFoundError as e:
        logger.error(f"Failed to load metrics for final results. Missing file: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error while printing final results: {e}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Hybrid URL Intelligence pipeline orchestrator")
    parser.add_argument("--mode", type=str, choices=['gnn', 'legacy'], default='gnn', help="Graph model mode ('gnn' or 'legacy')")
    parser.add_argument("--skip-prep", action="store_true", help="Skip data preparation (fix_data.py) if fixed dataset exists")
    parser.add_argument("--alpha", type=float, default=0.7, help="Alpha parameter for ensemble probability fusion")
    args = parser.parse_args()

    logger.info("Initializing Full Detection Pipeline")
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    fixed_dataset_path = os.path.join(base_dir, "data", "raw", "malicious_phish_fixed.csv")
    raw_dataset_path = os.path.join(base_dir, "data", "raw", "malicious_phish.csv")
    
    # 0. Optional Data Preparation (fix_data.py)
    run_prep = True
    if args.skip_prep and os.path.exists(fixed_dataset_path):
        logger.info("Cleaned dataset already exists and --skip-prep specified. Skipping data preparation.")
        run_prep = False
        
    if run_prep:
        fix_data_script = os.path.join(base_dir, "fix_data.py")
        run_step("Data Preparation (Bias Mitigation)", fix_data_script, base_dir)
        
    # Set the feature builder raw data path source
    builder_input = fixed_dataset_path if os.path.exists(fixed_dataset_path) else raw_dataset_path
    
    # 1. Feature Builder
    feature_builder_script = os.path.join(base_dir, "src", "feature_engineering", "feature_builder.py")
    feature_dataset_path = os.path.join(base_dir, "data", "processed", "feature_dataset.parquet")
    run_step("Feature Builder", feature_builder_script, base_dir, [
        "--raw-data-path", builder_input,
        "--output-path", feature_dataset_path
    ])
    
    # 2. Run LightGBM Training
    lightgbm_train_script = os.path.join(base_dir, "src", "models", "lightgbm_train.py")
    run_step("LightGBM Training", lightgbm_train_script, base_dir)
    
    # 3. Run Graph/GNN Step
    if args.mode == "gnn":
        graph_script = os.path.join(base_dir, "src", "graph", "gnn_train.py")
        run_step("GraphSAGE GNN Training", graph_script, base_dir)
        
        # Verify gnn features output
        gnn_features_path = os.path.join(base_dir, "data", "processed", "gnn_features.parquet")
        if not os.path.exists(gnn_features_path):
            logger.error(f"GNN features not found at {gnn_features_path} after GNN training step.")
            sys.exit(1)
        fusion_data_path = gnn_features_path
    else:
        graph_script = os.path.join(base_dir, "src", "graph", "domain_graph.py")
        run_step("Legacy Domain Graph Features", graph_script, base_dir)
        
        # Verify legacy graph features output
        graph_features_path = os.path.join(base_dir, "data", "processed", "graph_features.parquet")
        if not os.path.exists(graph_features_path):
            logger.error(f"Legacy graph features not found at {graph_features_path} after Domain Graph step.")
            sys.exit(1)
        fusion_data_path = graph_features_path
        
    # 4. Run Hybrid Fusion
    hybrid_fusion_script = os.path.join(base_dir, "src", "fusion", "hybrid_fusion.py")
    run_step("Hybrid Fusion", hybrid_fusion_script, base_dir, [
        "--data-path", fusion_data_path,
        "--alpha", str(args.alpha)
    ])
    
    # 5. Print final results
    print_final_results(base_dir)
    logger.info("Pipeline execution completed successfully.")

if __name__ == "__main__":
    main()
