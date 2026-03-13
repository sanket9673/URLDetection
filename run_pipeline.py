import os
import sys
import json
import subprocess
import logging

# orchestration entrypoint (runs feature generation → model training script → 
# -> graph features → fusion → prints final metrics).


try:
    from src.logger_config import get_logger
    logger = get_logger(__name__)
except ImportError:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

def run_step(step_name: str, script_path: str, base_dir: str):
    logger.info(f"--- Starting Step: {step_name} ---")
    try:
        # Run the scripts as a subprocess
        env = os.environ.copy()
        env["PYTHONPATH"] = base_dir
        result = subprocess.run([sys.executable, script_path], check=True, capture_output=True, text=True, cwd=base_dir, env=env)
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
    logger.info("Initializing Full Detection Pipeline")
    
    # Enforce executing from or knowing the base directory
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 1. Feature Builder
    feature_builder_script = os.path.join(base_dir, "src", "feature_engineering", "feature_builder.py")
    run_step("Feature Builder", feature_builder_script, base_dir)
    
    # 2. Run Training
    lightgbm_train_script = os.path.join(base_dir, "src", "models", "lightgbm_train.py")
    run_step("LightGBM Training", lightgbm_train_script, base_dir)
    
    # 3. Run Graph Module
    domain_graph_script = os.path.join(base_dir, "src", "graph", "domain_graph.py")
    run_step("Domain Graph Intelligence", domain_graph_script, base_dir)
    
    # 4. Run Fusion
    hybrid_fusion_script = os.path.join(base_dir, "src", "fusion", "hybrid_fusion.py")
    run_step("Hybrid Fusion", hybrid_fusion_script, base_dir)
    
    # 5. Print final results
    print_final_results(base_dir)
    
    logger.info("Pipeline execution completed successfully.")

if __name__ == "__main__":
    main()
