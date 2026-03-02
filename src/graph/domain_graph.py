import os
import time
import logging
import psutil
import pandas as pd
import numpy as np
import tldextract
from sklearn.model_selection import train_test_split

try:
    from src.logger_config import get_logger
    logger = get_logger(__name__)
except ImportError:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

class DomainGraphFeatureBuilder:
    """
    Computes domain-level intelligence safely ensuring no data leakage.
    Features:
    - domain_frequency
    - class_distribution per domain
    - domain_class_probability vector
    - tld_class_probability
    - domain_entropy
    """
    def __init__(self, data_path="data/processed/feature_dataset.parquet", 
                 output_path="data/processed/graph_features.parquet"):
        self.data_path = data_path
        self.output_path = output_path
        self.target_col = 'target'
        
    def log_memory(self, prefix=""):
        process = psutil.Process(os.getpid())
        mem_mb = process.memory_info().rss / (1024 ** 2)
        logger.info(f"{prefix} Memory Usage: {mem_mb:.2f} MB")

    def run(self):
        start_time = time.time()
        logger.info("Starting Domain Graph computations...")
        self.log_memory("Initial")

        # 1. Load Data
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Missing dataset at {self.data_path}")
            
        df = pd.read_parquet(self.data_path)
        
        if self.target_col not in df.columns:
            if 'label' in df.columns:
                self.target_col = 'label'
            else:
                raise ValueError("Target column not found in dataset.")
                
        logger.info(f"Loaded dataset for graph intelligence. Shape: {df.shape}")
        
        # 2. Extract domain using tldextract
        logger.info("Extracting domains and TLDs using tldextract...")
        domains = []
        tlds = []
        for u in df['url']:
            ext = tldextract.extract(str(u))
            # domain.suffix is the "registered domain"
            d = f"{ext.domain}.{ext.suffix}" if ext.domain else ext.suffix
            domains.append(d)
            tlds.append(ext.suffix)
            
        df['registered_domain'] = domains
        df['tld'] = tlds
        
        unique_domains = df['registered_domain'].nunique()
        logger.info(f"Total Unique Domains: {unique_domains}")
        self.log_memory("Post-Extraction")

        # 3. Simulate train-test split to prevent data leakage (must match lightgbm_train.py split)
        # Using the exact same stratify logic (test_size=0.30, random_state=42)
        logger.info("Isolating train set to prevent data leakage...")
        df_train, _ = train_test_split(
            df, test_size=0.30, stratify=df[self.target_col], random_state=42
        )
        logger.info(f"Train subset shape: {df_train.shape}")
        
        classes = sorted(df_train[self.target_col].unique())
        global_probs = df_train[self.target_col].value_counts(normalize=True).to_dict()
        
        # 4. Compute distributions and frequencies
        logger.info("Computing domain-level and TLD-level frequencies & class distributions...")
        
        # TLD distributions
        tld_counts = df_train['tld'].value_counts()
        tld_class_counts = pd.crosstab(df_train['tld'], df_train[self.target_col])
        for c in classes:
            if c not in tld_class_counts.columns:
                tld_class_counts[c] = 0
        tld_class_probs = tld_class_counts.div(tld_counts, axis=0)
        
        # Domain distributions
        domain_counts = df_train['registered_domain'].value_counts()
        domain_class_counts = pd.crosstab(df_train['registered_domain'], df_train[self.target_col])
        for c in classes:
            if c not in domain_class_counts.columns:
                domain_class_counts[c] = 0
        domain_class_probs = domain_class_counts.div(domain_counts, axis=0)
        
        # Domain Entropy
        def compute_entropy(row):
            p = row.values
            p = p[p > 0]
            if len(p) == 0:
                return 0.0
            return -np.sum(p * np.log2(p))
            
        domain_entropy = domain_class_probs.apply(compute_entropy, axis=1)
        
        # Convert to dict for fast mapping row-by-row
        tld_prob_dict = tld_class_probs.to_dict('index')
        domain_prob_dict = domain_class_probs.to_dict('index')
        domain_count_dict = domain_counts.to_dict()
        domain_entropy_dict = domain_entropy.to_dict()
        domain_class_cnts_dict = domain_class_counts.to_dict('index')
        
        # 5. Merge back to URL level safely
        logger.info("Mapping computed graph scores back to full URL dataset...")
        
        freqs = []
        entropies = []
        domain_class_counts_arr = {c: [] for c in classes}
        graph_scores = {c: [] for c in classes}
        tld_scores = {c: [] for c in classes}
        
        for idx, row in df.iterrows():
            d = row['registered_domain']
            t = row['tld']
            
            # Unseen domains in training map to 0 frequency/entropy/counts
            freqs.append(domain_count_dict.get(d, 0))
            entropies.append(domain_entropy_dict.get(d, 0.0))
            
            d_probs = domain_prob_dict.get(d, None)
            t_probs = tld_prob_dict.get(t, None)
            d_cnts = domain_class_cnts_dict.get(d, None)
            
            for c in classes:
                # Fallback to global probability if TLD unseen
                tld_val = t_probs[c] if t_probs else global_probs.get(c, 0.0)
                tld_scores[c].append(tld_val)
                
                # Fallback to TLD probability if Domain unseen
                d_val = d_probs[c] if d_probs else tld_val
                graph_scores[c].append(d_val)
                
                # Zero counts if unseen
                cnt_val = d_cnts[c] if d_cnts else 0
                domain_class_counts_arr[c].append(cnt_val)
                
        df['domain_frequency'] = freqs
        df['domain_entropy'] = entropies
        
        for c in classes:
            df[f'domain_class_{c}_count'] = domain_class_counts_arr[c]
            df[f'tld_class_{c}_prob'] = tld_scores[c]
            df[f'domain_class_{c}_prob'] = graph_scores[c]
            # Graph Score (Alias of domain class prob)
            df[f'R_graph_class_{c}'] = graph_scores[c]
            
        # Optional: cleanup non-numerical temporary columns if needed
        # df.drop(columns=['registered_domain', 'tld'], inplace=True)
        
        logger.info(f"Graph feature generation complete. Final Columns: {df.shape[1]}")
        self.log_memory("Pre-Save")
        
        # 6. Save Data
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        df.to_parquet(self.output_path, engine='pyarrow', index=False)
        logger.info(f"Successfully saved graph features to {self.output_path}")
        
        execution_time = time.time() - start_time
        logger.info(f"Domain Graph computations finished in {execution_time:.2f} seconds.")
        self.log_memory("Final")
        
        return df

if __name__ == "__main__":
    builder = DomainGraphFeatureBuilder()
    builder.run()
