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
    
    # Save Graph Data and Mappings for Inference Dashboard
    os.makedirs("models", exist_ok=True)
    import pickle
    with open("models/gnn_mappings.pkl", "wb") as f:
        pickle.dump({"domain_mapping": domain_mapping, "tld_mapping": tld_mapping, "feature_cols": feature_cols}, f)
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
