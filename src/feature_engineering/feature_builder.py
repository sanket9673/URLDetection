import os
import time
import logging
import numpy as np
import pandas as pd
from src.logger_config import logger

class FeatureBuilder:
    """
    Feature Builder for extracting numerical lexical features from URLs.
    Extracts 35+ structural, statistical, and suspicious features.
    """
    def __init__(self, raw_data_path: str, output_path: str):
        self.raw_data_path = raw_data_path
        self.output_path = output_path
        
        # Target encoding mapping
        self.type_mapping = {
            'benign': 0,
            'defacement': 1,
            'phishing': 2,
            'malware': 3
        }
        
    def load_data(self) -> pd.DataFrame:
        """
        Loads CSV from raw data with memory error fallback to chunking.
        """
        logger.info(f"Loading raw data from {self.raw_data_path}")
        try:
            df = pd.read_csv(self.raw_data_path)
            logger.info("Successfully loaded data entirely.")
        except MemoryError:
            logger.warning("MemoryError encountered. Falling back to chunk loading.")
            chunks = []
            for chunk in pd.read_csv(self.raw_data_path, chunksize=50000):
                chunks.append(chunk)
            df = pd.concat(chunks, ignore_index=True)
            logger.info("Successfully loaded data in chunks.")
        except Exception as e:
            logger.error(f"Failed to load data: {e}")
            raise e
            
        return df

    def validate_and_clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Validates columns and cleans malformed URLs.
        """
        if 'url' not in df.columns or 'type' not in df.columns:
            logger.error("Required columns ('url', 'type') not found in dataset")
            raise ValueError("Columns 'url' and 'type' must be present.")
            
        initial_shape = df.shape
        df = df.dropna(subset=['url', 'type'])
        df['url'] = df['url'].astype(str)
        # remove empty strings
        df = df[df['url'].str.strip() != '']
        logger.info(f"Cleaned malformed URLs. Shape changed from {initial_shape} to {df.shape}")
        return df.copy()

    def build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extracts structural, statistical, and suspicious features
        using vectorized pandas operations (NO LOOPS).
        """
        logger.info("Extracting features using vectorized operations.")
        
        # Using regex to capture domain and path
        domain_extracted = df['url'].str.extract(r'^(?:https?://)?([^/:\?]+)')[0].fillna('')
        path_extracted = df['url'].str.extract(r'^(?:https?://)?[^/:\?]+(/[^\?]*)')[0].fillna('')
        
        # ----------- STRUCTURAL (1. Required + Additional) -----------
        logger.info("Building STRUCTURAL features.")
        
        df['url_length'] = df['url'].str.len()
        df['domain_length'] = domain_extracted.str.len()
        df['path_length'] = path_extracted.str.len()
        df['subdomain_count'] = domain_extracted.str.count(r'\.')
        df['num_dots'] = df['url'].str.count(r'\.')
        df['num_digits'] = df['url'].str.count(r'\d')
        df['num_hyphens'] = df['url'].str.count(r'-')
        df['num_special_chars'] = df['url'].str.count(r'[^a-zA-Z0-9.\-/:\?]')
        df['https_flag'] = df['url'].str.startswith('https').astype(int)
        df['contains_ip'] = domain_extracted.str.contains(r'\b(?:\d{1,3}\.){3}\d{1,3}\b').astype(int)
        
        # Additional structural
        df['num_underscores'] = df['url'].str.count(r'_')
        df['num_equals'] = df['url'].str.count(r'=')
        df['num_ampersands'] = df['url'].str.count(r'&')
        df['num_percent'] = df['url'].str.count(r'%')
        df['num_semicolons'] = df['url'].str.count(r';')
        df['num_tilde'] = df['url'].str.count(r'~')
        df['num_plus'] = df['url'].str.count(r'\+')
        df['num_asterisk'] = df['url'].str.count(r'\*')
        df['num_hash'] = df['url'].str.count(r'#')
        df['vowel_count'] = df['url'].str.count(r'[aeiouAEIOU]')
        df['consonant_count'] = df['url'].str.count(r'[bcdfghjklmnpqrstvwxyzBCDFGHJKLMNPQRSTVWXYZ]')
        df['letter_count'] = df['url'].str.count(r'[a-zA-Z]')
        df['has_port'] = df['url'].str.contains(r':\d+').astype(int)

        # ----------- STATISTICAL (2. Required + Additional) -----------
        logger.info("Building STATISTICAL features.")
        
        # Entropy vectorized using apply
        def compute_entropy(s):
            if not s: return 0.0
            _, counts = np.unique(list(s), return_counts=True)
            probs = counts / len(s)
            return -np.sum(probs * np.log2(probs))

        df['entropy'] = df['url'].apply(compute_entropy).astype(float)
        df['domain_entropy'] = domain_extracted.apply(compute_entropy).astype(float)
        
        # unique_char_ratio
        df['unique_char_ratio'] = df['url'].apply(lambda x: len(set(x)) if isinstance(x, str) else 0) / df['url_length']
        df['unique_char_ratio'] = df['unique_char_ratio'].fillna(0)
        
        # required ratios
        df['digit_ratio'] = (df['num_digits'] / df['url_length']).fillna(0)
        df['special_char_ratio'] = (df['num_special_chars'] / df['url_length']).fillna(0)
        
        # query_param_count
        df['query_param_count'] = df['url'].str.count(r'[&?]')
        
        # longest_token_length
        token_lengths = df['url'].str.split(r'\W+').apply(
            lambda x: max((len(t) for t in x if t), default=0) if isinstance(x, list) else 0
        )
        df['longest_token_length'] = token_lengths
        
        # Additional statistical
        df['vowel_ratio'] = (df['vowel_count'] / df['url_length']).fillna(0)
        df['consonant_ratio'] = (df['consonant_count'] / df['url_length']).fillna(0)
        df['letter_ratio'] = (df['letter_count'] / df['url_length']).fillna(0)

        # ----------- SUSPICIOUS (3. Required + Additional) -----------
        logger.info("Building SUSPICIOUS features.")
        suspicious_keywords = r'(?i)(login|verify|update|secure|bank|account|confirm|free|bonus|admin|webscr|paypal|signin)'
        df['suspicious_keyword_count'] = df['url'].str.count(suspicious_keywords)
        df['contains_at'] = df['url'].str.contains(r'@').astype(int)
        df['double_slash_count'] = df['url'].str.count(r'//') - 1  # Excluding the http:// prefix
        df['double_slash_count'] = df['double_slash_count'].clip(lower=0)
        
        # Additional suspicious
        df['is_shortened'] = domain_extracted.str.contains(r'(bit\.ly|goo\.gl|t\.co|ow\.ly|is\.gd|tinyurl|tr\.im)').astype(int)
        df['has_exe_or_zip'] = df['url'].str.contains(r'(?i)(\.exe|\.zip|\.rar|\.tar|\.gz|\.7z)').astype(int)
        df['suspicious_tld'] = domain_extracted.str.contains(r'(?i)(\.top|\.xyz|\.club|\.site|\.online|\.pro|\.pw)').astype(int)
        df['multiple_subdomains'] = (df['subdomain_count'] > 2).astype(int)

        return df

    def encode_target(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Encodes target multiclass labels.
        """
        logger.info("Encoding target class labels.")
        df['type'] = df['type'].str.lower().str.strip()
        df['target'] = df['type'].map(self.type_mapping)
        
        unmapped_mask = df['target'].isna()
        if unmapped_mask.any():
            logger.warning(f"Dropping {unmapped_mask.sum()} rows with unknown type labels.")
            df = df[~unmapped_mask]
        
        df['target'] = df['target'].astype(int)
        return df

    def save_features(self, df: pd.DataFrame):
        """
        Saves dataset to parquet.
        """
        output_dir = os.path.dirname(self.output_path)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
            
        try:
            df.to_parquet(self.output_path, engine='pyarrow', index=False)
            logger.info(f"Successfully saved features to {self.output_path}")
        except Exception as e:
            logger.error(f"Error saving to parquet: {e}")
            raise e

    def run(self) -> pd.DataFrame:
        """
        Runs the full feature engineering pipeline.
        """
        start_time = time.time()
        logger.info("Starting feature engineering pipeline.")
        
        try:
            df = self.load_data()
            df = self.validate_and_clean(df)
            df = self.build_features(df)
            df = self.encode_target(df)
            
            self.save_features(df)
            
            # Detailed Logging
            time_taken = time.time() - start_time
            logger.info("=================================")
            logger.info("Feature Engineering Summary")
            logger.info(f"Time taken       : {time_taken:.2f} seconds")
            logger.info(f"Dataset shape    : {df.shape}")
            feature_cols = [c for c in df.columns if c not in ['url', 'type', 'target']]
            logger.info(f"Feature count    : {len(feature_cols)}")
            
            distribution = df['target'].value_counts().to_dict()
            inv_map = {v: k for k, v in self.type_mapping.items()}
            logger.info("Class distribution:")
            for k, v in distribution.items():
                logger.info(f"  {inv_map[k]}: {v}")
            logger.info("=================================")
            
            return df
            
        except Exception as e:
            logger.error(f"Feature engineering pipeline failed: {e}")
            raise e

if __name__ == "__main__":
    import yaml
    
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    config_path = os.path.join(base_dir, 'config', 'config.yaml')
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        raw_data_path = os.path.join(base_dir, config['data']['raw_dataset'])
        output_path = os.path.join(base_dir, config['data']['processed_dataset'].replace('.csv', '.parquet'))
    except Exception as e:
        logger.warning(f"Could not load config defaults: {e}. Using local paths.")
        raw_data_path = "data/raw/malicious_phish.csv"
        output_path = "data/processed/feature_dataset.parquet"
        
    builder = FeatureBuilder(raw_data_path=raw_data_path, output_path=output_path)
    df = builder.run()
