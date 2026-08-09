import os
import time
import json
import logging
import tldextract
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.nn import SAGEConv, to_hetero
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

try:
    from src.logger_config import get_logger
    logger = get_logger(__name__)
except ImportError:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

class BaseSAGE(torch.nn.Module):
    """
    Base Message Passing Networks using GraphSAGE.
    We will wrap this in `to_hetero` to dynamically adapt to the
    URL -> Domain -> TLD heterogeneous bipartite structure.
    """
    def __init__(self, hidden_channels):
        super().__init__()
        # 2 layers of neighbor sampling and aggregation
        self.conv1 = SAGEConv((-1, -1), hidden_channels)
        self.conv2 = SAGEConv((-1, -1), hidden_channels)
        
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index).relu()
        return x

class HeteroGraphSAGE(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels, metadata):
        super().__init__()
        # Multi-layer GraphSAGE architecture
        self.gnn = BaseSAGE(hidden_channels)
        # Adapt to heterogeneous structure dynamically
        self.gnn = to_hetero(self.gnn, metadata, aggr='mean')
        
        # Dedicated MLP for URL nodes
        self.lin1 = torch.nn.Linear(hidden_channels, hidden_channels)
        # The prompt instructed: Linear -> ReLU -> Dropout -> Linear -> Softmax
        self.lin2 = torch.nn.Linear(hidden_channels, out_channels)
        
    def forward(self, x_dict, edge_index_dict):
        # 1. Neighbor sampling and aggregation across Heterogeneous graphs
        node_embs = self.gnn(x_dict, edge_index_dict)
        
        # 2. Extract final URL embeddings
        url_emb = node_embs['url']
        
        # 3. Final MLP (Linear -> ReLU -> Dropout -> Linear -> Softmax)
        x = self.lin1(url_emb).relu()
        x = F.dropout(x, p=0.5, training=self.training)
        logits = self.lin2(x)
        
        # Output 4-class probability vector as required
        probs = F.softmax(logits, dim=1)
        return probs

def prepare_hetero_graph(df_path="data/processed/feature_dataset.parquet", target_col="target"):
    logger.info("Building Heterogeneous Graph Configuration...")
    df = pd.read_parquet(df_path)
    
    # Identify target column correctly
    if target_col not in df.columns and 'label' in df.columns:
        target_col = 'label'
        
    # Exclude non-feature columns for URL nodes
    exclude_cols = ['url', 'type', target_col, 'label']
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    
    # 1. Extract Domains and TLDs
    domains = []
    tlds = []
    for u in df['url']:
        ext = tldextract.extract(str(u))
        dom = f"{ext.domain}.{ext.suffix}" if ext.domain else ext.suffix
        domains.append(dom)
        tlds.append(ext.suffix)
        
    df['registered_domain'] = domains
    df['tld'] = tlds
    
    # 2. Assign Continuous Integer Node IDs
    domain_mapping = {d: i for i, d in enumerate(df['registered_domain'].unique())}
    tld_mapping = {t: i for i, t in enumerate(df['tld'].unique())}
    url_mapping = {u: i for i, u in enumerate(df['url'])}  # Preserving exact 0-N indexing
    
    num_urls = len(df)
    num_domains = len(domain_mapping)
    num_tlds = len(tld_mapping)
    
    # 3. Form URL Features
    url_features = torch.tensor(df[feature_cols].values, dtype=torch.float)
    
    # 4. Form Domain Features (Aggregated Lexical Features of Connected URLs)
    logger.info("Aggregating URL traits for Domain Embeddings...")
    domain_features_np = np.zeros((num_domains, len(feature_cols)))
    domain_counts = np.zeros((num_domains, 1))
    url_feats_np = df[feature_cols].values
    
    for i, dom in enumerate(df['registered_domain']):
        d_idx = domain_mapping[dom]
        domain_features_np[d_idx] += url_feats_np[i]
        domain_counts[d_idx] += 1
        
    domain_features_np = domain_features_np / np.where(domain_counts == 0, 1, domain_counts)
    domain_features = torch.tensor(domain_features_np, dtype=torch.float)
    
    # 5. Form TLD Features (One-Hot Encoded strings)
    logger.info("Encoding TLD strings via One-Hot vectors...")
    tld_features = torch.eye(num_tlds, dtype=torch.float)
    
    # 6. Build Edges (Bipartite Structure)
    # URL -> (belongs_to) -> Domain
    url_domain_src = []
    url_domain_dst = []
    
    # Domain -> (belongs_to) -> TLD
    domain_tld_src = []
    domain_tld_dst = []
    
    added_domain_tld = set()
    
    for _, row in df.iterrows():
        u_idx = url_mapping[row['url']]
        d_idx = domain_mapping[row['registered_domain']]
        t_idx = tld_mapping[row['tld']]
        
        url_domain_src.append(u_idx)
        url_domain_dst.append(d_idx)
        
        pair = (d_idx, t_idx)
        if pair not in added_domain_tld:
            domain_tld_src.append(pair[0])
            domain_tld_dst.append(pair[1])
            added_domain_tld.add(pair)
            
    edge_index_url_domain = torch.tensor([url_domain_src, url_domain_dst], dtype=torch.long)
    edge_index_domain_tld = torch.tensor([domain_tld_src, domain_tld_dst], dtype=torch.long)
    
    # 7. Construct PyG HeteroData Object
    data = HeteroData()
    data['url'].x = url_features
    data['domain'].x = domain_features
    data['tld'].x = tld_features
    
    data['url', 'belongs_to', 'domain'].edge_index = edge_index_url_domain
    data['domain', 'belongs_to', 'tld'].edge_index = edge_index_domain_tld
    
    import torch_geometric.transforms as T
    data = T.ToUndirected()(data)
    
    # 8. Manage Train/Val/Test Masking carefully (70% Train, prevent data leakage)
    idx = np.arange(num_urls)
    labels = df[target_col].values
    
    train_idx, temp_idx, y_train, y_temp = train_test_split(
        idx, labels, test_size=0.30, stratify=labels, random_state=42
    )
    val_idx, test_idx, _, _ = train_test_split(
        temp_idx, y_temp, test_size=0.50, stratify=y_temp, random_state=42
    )
    
    train_mask = torch.zeros(num_urls, dtype=torch.bool)
    val_mask = torch.zeros(num_urls, dtype=torch.bool)
    test_mask = torch.zeros(num_urls, dtype=torch.bool)
    
    train_mask[train_idx] = True
    val_mask[val_idx] = True
    test_mask[test_idx] = True
    
    data['url'].train_mask = train_mask
    data['url'].val_mask = val_mask
    data['url'].test_mask = test_mask
    
    # Encode Target Labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(labels)
    data['url'].y = torch.tensor(y_encoded, dtype=torch.long)
    
    # Compute global domain reputation averages strictly on the training set (to prevent data leakage)
    train_df = df.iloc[train_idx]
    train_domain_names = train_df['registered_domain'].unique()
    train_domain_ids = [domain_mapping[d] for d in train_domain_names]
    global_domain_reputation_avg = domain_features_np[train_domain_ids].mean(axis=0)

    # Save Graph Data and Mappings for Inference Dashboard
    os.makedirs("models", exist_ok=True)
    import pickle
    with open("models/gnn_mappings.pkl", "wb") as f:
        pickle.dump({
            "domain_mapping": domain_mapping,
            "tld_mapping": tld_mapping,
            "feature_cols": feature_cols,
            "global_domain_reputation_avg": global_domain_reputation_avg
        }, f)
    torch.save(data, "models/gnn_graph_data.pt")
    
    logger.info(f"Graph Construction Complete. URL Nodes: {num_urls}, Domain Nodes: {num_domains}, TLD Nodes: {num_tlds}")
    
    return data, le, df

def train_gnn():
    start_time = time.time()
    logger.info("Starting GraphSAGE training sequence...")
    
    # Get Data
    data, le, df = prepare_hetero_graph()
    
    # Set PyTorch backend device
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    logger.info(f"Using compute device: {device}")
    
    data = data.to(device)
    
    # Instantiate Model
    hidden_channels = 64
    out_channels = len(le.classes_)
    
    model = HeteroGraphSAGE(hidden_channels=hidden_channels, 
                            out_channels=out_channels, 
                            metadata=data.metadata()).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)
    
    # Training Loop directly handling Softmax Probabilities
    # We use NLLLoss after log() to correctly train via CrossEntropy logic while outputting softmax probs
    criterion = torch.nn.NLLLoss()
    
    epochs = 150
    best_val_loss = float('inf')
    
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        
        out_probs = model(data.x_dict, data.edge_index_dict)
        # Log probability output to interface with NLLLoss properly
        log_probs = torch.log(out_probs + 1e-9)
        loss = criterion(log_probs[data['url'].train_mask], data['url'].y[data['url'].train_mask])
        
        loss.backward()
        optimizer.step()
        
        # Validation Evaluation
        model.eval()
        with torch.no_grad():
            val_out = model(data.x_dict, data.edge_index_dict)
            val_log_probs = torch.log(val_out + 1e-9)
            val_loss = criterion(val_log_probs[data['url'].val_mask], data['url'].y[data['url'].val_mask])
            
            val_pred = val_out.argmax(dim=1)
            val_correct = (val_pred[data['url'].val_mask] == data['url'].y[data['url'].val_mask]).sum()
            val_acc = int(val_correct) / int(data['url'].val_mask.sum())
            
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            # Save the optimal weights
            os.makedirs("models", exist_ok=True)
            # Use weights_only=True during save is invalid for torch.save, valid on load
            torch.save(model.state_dict(), "models/graphsage_model.pth")
            
        if epoch % 25 == 0:
            logger.info(f'Epoch: {epoch:03d}, Training Loss: {loss:.4f}, Validation Loss: {val_loss:.4f}, Validation Acc: {val_acc:.4f}')

    logger.info("Training complete. Computing and exporting Test probabilities...")
    
    # Reload best model weights and calculate predictions
    model.load_state_dict(torch.load("models/graphsage_model.pth", weights_only=True))
    model.eval()
    
    with torch.no_grad():
        final_probs = model(data.x_dict, data.edge_index_dict)
    
    # Dump test-set GNN features to standard Parquet format for fusion
    # To retain backwards-compatibility without breaking hybrid_fusion assumptions,
    # we simulate the same structure by mapping back exactly to all nodes,
    # appending the probabilities directly to the original dataframe alignment.
    
    probs_cpu = final_probs.cpu().numpy()
    
    # Save target specific class names from mapping (Alphabetical alignment)
    classes = le.classes_
    for i, cls in enumerate(classes):
        # We save probabilities per URL as "R_graph_class_X" or "domain_class_X_prob" 
        # as expected in hybrid_fusion.py
        df[f'domain_class_{cls}_prob'] = probs_cpu[:, i]
    
    # We retain all original features (e.g. url_length) so hybrid_fusion.py can still run LightGBM on them
    save_df = df
    
    os.makedirs("data/processed", exist_ok=True)
    out_path = "data/processed/gnn_features.parquet"
    save_df.to_parquet(out_path, engine='pyarrow', index=False)
    
    with open("models/gnn_classes.json", "w") as f:
        json.dump([int(c) if isinstance(c, (np.integer, int)) else str(c) for c in classes], f)
        
    logger.info(f"GraphSAGE model saved to models/graphsage_model.pth and GNN predictions to {out_path}.")
    logger.info(f"Execution took {time.time() - start_time:.2f} seconds.")

if __name__ == '__main__':
    train_gnn()


# ==========================================
# DYNAMIC INFERENCE & COLD-START INJECTION
# ==========================================

import hashlib
import threading

gnn_lock = threading.Lock()

def get_tld_ngram_embedding(tld_str: str, num_tlds: int) -> np.ndarray:
    """
    Computes a deterministic character n-gram mean embedding of a TLD string
    to map unseen/cold-start TLDs to the feature space.
    """
    tld_str = tld_str.strip('.')
    if not tld_str:
        return np.zeros(num_tlds)
    
    # Extract 2-grams and 3-grams
    ngrams = []
    for i in range(len(tld_str) - 1):
        ngrams.append(tld_str[i:i+2])
    for i in range(len(tld_str) - 2):
        ngrams.append(tld_str[i:i+3])
        
    # Fallback to characters if TLD is extremely short
    if not ngrams:
        ngrams = list(tld_str)
        
    feat = np.zeros(num_tlds)
    for ngram in ngrams:
        # Use hashlib for deterministic hashing across sessions
        h = int(hashlib.md5(ngram.encode('utf-8')).hexdigest(), 16)
        idx = h % num_tlds
        feat[idx] += 1.0
        
    # Normalize to get mean embedding
    if feat.sum() > 0:
        feat = feat / feat.sum()
    return feat

def predict_gnn_dynamic(urls: list, df_features: pd.DataFrame, gnn_model, gnn_data, gnn_mappings):
    """
    Dynamically injects unseen URLs, domains, and TLDs into the graph,
    executes GraphSAGE prediction, and rolls back the graph state to isolate memory.
    Thread-safe via gnn_lock.
    """
    with gnn_lock:
        device = next(gnn_model.parameters()).device
        feature_cols = gnn_mappings['feature_cols']
        domain_mapping = gnn_mappings['domain_mapping']
        tld_mapping = gnn_mappings['tld_mapping']
        
        # Determine global domain reputation fallback
        if 'global_domain_reputation_avg' in gnn_mappings:
            global_domain_reputation_avg = gnn_mappings['global_domain_reputation_avg']
        else:
            # Fallback if pickle is not yet updated
            global_domain_reputation_avg = gnn_data['domain'].x.mean(dim=0).cpu().numpy()
            
        # Parse lexical features for the input URLs
        available_feats = []
        for idx in range(len(urls)):
            row_feats = []
            for c in feature_cols:
                if c in df_features.columns:
                    row_feats.append(df_features[c].values[idx])
                else:
                    row_feats.append(0.0)
            available_feats.append(row_feats)
        url_feat_tensor = torch.tensor(available_feats, dtype=torch.float).to(device)
        
        # Save original graph state
        orig_url_x = gnn_data['url'].x
        orig_domain_x = gnn_data['domain'].x
        orig_tld_x = gnn_data['tld'].x
        
        orig_ud_edges = gnn_data['url', 'belongs_to', 'domain'].edge_index
        orig_dt_edges = gnn_data['domain', 'belongs_to', 'tld'].edge_index
        orig_du_edges = gnn_data['domain', 'rev_belongs_to', 'url'].edge_index
        orig_td_edges = gnn_data['tld', 'rev_belongs_to', 'domain'].edge_index
        
        # Append URL features
        gnn_data['url'].x = torch.cat([gnn_data['url'].x, url_feat_tensor], dim=0)
        
        orig_url_count = orig_url_x.shape[0]
        orig_domain_count = orig_domain_x.shape[0]
        orig_tld_count = orig_tld_x.shape[0]
        
        temp_domain_mapping = domain_mapping.copy()
        temp_tld_mapping = tld_mapping.copy()
        
        new_domain_feats = []
        new_tld_feats = []
        
        new_ud_src, new_ud_dst = [], []
        new_du_src, new_du_dst = [], []
        new_dt_src, new_dt_dst = [], []
        new_td_src, new_td_dst = [], []
        
        added_dt_pairs = set()
        
        # Dynamic extraction and mapping
        for idx, url in enumerate(urls):
            u_idx = orig_url_count + idx
            
            ext = tldextract.extract(url)
            domain = f"{ext.domain}.{ext.suffix}" if ext.domain else ext.suffix
            tld = ext.suffix
            
            # Resolve domain node ID
            if domain in temp_domain_mapping:
                d_idx = temp_domain_mapping[domain]
            else:
                d_idx = orig_domain_count + len(new_domain_feats)
                temp_domain_mapping[domain] = d_idx
                # Cold-start fallback: use global domain reputation average
                new_domain_feats.append(global_domain_reputation_avg)
                
            # Resolve TLD node ID
            if tld in temp_tld_mapping:
                t_idx = temp_tld_mapping[tld]
            else:
                t_idx = orig_tld_count + len(new_tld_feats)
                temp_tld_mapping[tld] = t_idx
                # Cold-start fallback: compute deterministic character n-gram mean embedding
                tld_emb = get_tld_ngram_embedding(tld, orig_tld_count)
                new_tld_feats.append(tld_emb)
                
            # Map edges (URL -> belongs_to -> Domain)
            new_ud_src.append(u_idx)
            new_ud_dst.append(d_idx)
            new_du_src.append(d_idx)
            new_du_dst.append(u_idx)
            
            # Map edges (Domain -> belongs_to -> TLD)
            dt_pair = (d_idx, t_idx)
            if dt_pair not in added_dt_pairs:
                new_dt_src.append(d_idx)
                new_dt_dst.append(t_idx)
                new_td_src.append(t_idx)
                new_td_dst.append(d_idx)
                added_dt_pairs.add(dt_pair)
                
        # Update node feature tensors in HeteroData
        if new_domain_feats:
            new_domain_feats_tensor = torch.tensor(np.array(new_domain_feats), dtype=torch.float).to(device)
            gnn_data['domain'].x = torch.cat([gnn_data['domain'].x, new_domain_feats_tensor], dim=0)
        if new_tld_feats:
            new_tld_feats_tensor = torch.tensor(np.array(new_tld_feats), dtype=torch.float).to(device)
            gnn_data['tld'].x = torch.cat([gnn_data['tld'].x, new_tld_feats_tensor], dim=0)
            
        # Update edge index tensors in HeteroData
        new_ud_tensor = torch.tensor([new_ud_src, new_ud_dst], dtype=torch.long).to(device)
        gnn_data['url', 'belongs_to', 'domain'].edge_index = torch.cat([orig_ud_edges, new_ud_tensor], dim=1)
        
        new_du_tensor = torch.tensor([new_du_src, new_du_dst], dtype=torch.long).to(device)
        gnn_data['domain', 'rev_belongs_to', 'url'].edge_index = torch.cat([orig_du_edges, new_du_tensor], dim=1)
        
        if new_dt_src:
            new_dt_tensor = torch.tensor([new_dt_src, new_dt_dst], dtype=torch.long).to(device)
            gnn_data['domain', 'belongs_to', 'tld'].edge_index = torch.cat([orig_dt_edges, new_dt_tensor], dim=1)
            
            new_td_tensor = torch.tensor([new_td_src, new_td_dst], dtype=torch.long).to(device)
            gnn_data['tld', 'rev_belongs_to', 'domain'].edge_index = torch.cat([orig_td_edges, new_td_tensor], dim=1)
            
        # Run forward pass through GraphSAGE
        try:
            with torch.no_grad():
                probs = gnn_model(gnn_data.x_dict, gnn_data.edge_index_dict)
                # Extracted URL node predictions (last len(urls) elements)
                predictions = probs[-len(urls):].cpu().numpy()
        finally:
            # Enforce rollback of GNN data graph to isolate session states
            gnn_data['url'].x = orig_url_x
            gnn_data['domain'].x = orig_domain_x
            gnn_data['tld'].x = orig_tld_x
            
            gnn_data['url', 'belongs_to', 'domain'].edge_index = orig_ud_edges
            gnn_data['domain', 'rev_belongs_to', 'url'].edge_index = orig_du_edges
            gnn_data['domain', 'belongs_to', 'tld'].edge_index = orig_dt_edges
            gnn_data['tld', 'rev_belongs_to', 'domain'].edge_index = orig_td_edges
            
        # Re-normalize just to ensure sum(probs) = 1.0 safely
        predictions = predictions / (predictions.sum(axis=1, keepdims=True) + 1e-9)
        return predictions

