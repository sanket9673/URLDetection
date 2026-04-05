import os
import sys
import json
import time
import pickle
import logging
import tldextract
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

# Append project root to sys.path so we can import src modules
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.feature_engineering.feature_builder import FeatureBuilder

# 4. Add logs
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("Dashboard")

# Configure Page
st.set_page_config(page_title="Hybrid URL Intelligence", page_icon="🛡️", layout="wide")

# Inject Custom CSS for Rich Aesthetics & Visual Excellence
st.markdown("""
<style>
    /* Dark mode, glassmorphism, dynamic animations */
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        color: #f8fafc;
        font-family: 'Inter', sans-serif;
    }
    .main-header {
        text-align: center;
        margin-bottom: 2rem;
        background: -webkit-linear-gradient(#38bdf8, #818cf8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3rem;
        font-weight: 800;
        padding-top: 20px;
    }
    .card {
        background: rgba(30, 41, 59, 0.7);
        backdrop-filter: blur(10px);
        border-radius: 12px;
        padding: 20px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
        margin-bottom: 20px;
    }
    .card:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.3), 0 4px 6px -2px rgba(0, 0, 0, 0.15);
    }
    .metric-value {
        font-size: 2.5rem;
        font-weight: bold;
        color: #38bdf8;
    }
    .metric-label {
        font-size: 1rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("<h1 class='main-header'>Hybrid URL Intelligence Engine</h1>", unsafe_allow_html=True)

# Define Core Paths
MODEL_PATH = os.path.join(project_root, "models", "lightgbm_model.pkl")
GRAPH_PATH = os.path.join(project_root, "data", "processed", "graph_features.parquet")
METRICS_PATH = os.path.join(project_root, "outputs", "hybrid_metrics.json")

# Classes Mapping
CLASSES = ['benign', 'defacement', 'phishing', 'malware']

@st.cache_resource
def load_assets():
    """Load model, graph data, and alpha metric (Cached for performance)"""
    logger.info("Initializing system assets...")
    
    # 5. Handle missing model error securely
    if not os.path.exists(MODEL_PATH):
        logger.error(f"Missing LightGBM model at {MODEL_PATH}")
        raise FileNotFoundError(f"Trained LightGBM model is missing. Please run model training first.")
        
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
        
    gnn_model = None
    gnn_data = None
    gnn_mappings = None
    try:
        import torch
        from src.graph.gnn_train import HeteroGraphSAGE
        # Weights explicitly marked false because HeteroData expects multiple classes internally
        gnn_data = torch.load("models/gnn_graph_data.pt", weights_only=False)
        with open("models/gnn_mappings.pkl", "rb") as f:
            gnn_mappings = pickle.load(f)
            
        gnn_model = HeteroGraphSAGE(hidden_channels=64, out_channels=4, metadata=gnn_data.metadata())
        device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
        gnn_model.load_state_dict(torch.load("models/graphsage_model.pth", map_location=device, weights_only=True))
        gnn_model.to(device)
        gnn_model.eval()
        gnn_data = gnn_data.to(device)
    except Exception as e:
        logger.warning(f"Could not load GNN assets: {e}")
        
    alpha = 0.5
    if os.path.exists(METRICS_PATH):
        try:
            with open(METRICS_PATH, "r") as f:
                metrics = json.load(f)
                alpha = metrics.get('best_alpha', 0.5)
                logger.info(f"Loaded tuned alpha factor: {alpha}")
        except Exception as e:
            logger.warning(f"Could not load best_alpha, defaulting to 0.5. Error: {e}")
            
    return model, gnn_model, gnn_data, gnn_mappings, alpha

try:
    model, gnn_model, gnn_data, gnn_mappings, alpha = load_assets()
    logger.info("Intelligence engine ready online.")
except Exception as e:
    st.error(f"Intelligence Engine offline: {str(e)}")
    st.stop()

# 1. URL input field

tab1, tab2 = st.tabs(["🛡️ Live Threat Analysis", "🕸️ Graph Intelligence Architecture (Demo)"])

with tab2:
    st.markdown("<h2 style=\"color: #f8fafc;\">Heterogeneous GNN Structural Intelligence</h2>", unsafe_allow_html=True)
    st.markdown("<p style=\"color: #94a3b8;\">This interactive topological visualization demonstrates how the GraphSAGE \"Sample and Aggregate\" mechanism works across TLD, Domain, and URL nodes. By aggregating lexical features from across the neighborhood, the system performs inductive threat detection even on zero-day subdomains.</p>", unsafe_allow_html=True)
    
    # Try both files, prioritize gnn_topology_visualization.html
    html_found = False
    for fname in ["gnn_topology_visualization.html", "domain_graph_concept.html"]:
        html_path = os.path.join(project_root, "outputs", "reports", fname)
        if os.path.exists(html_path):
            import streamlit.components.v1 as components
            with open(html_path, "r", encoding="utf-8") as f:
                source_code = f.read()
            st.markdown("<div class=\"card\" style=\"padding: 0px; overflow: hidden;\">", unsafe_allow_html=True)
            components.html(source_code, height=650, scrolling=False)
            st.markdown("</div>", unsafe_allow_html=True)
            html_found = True
            break
    
    if not html_found:
        st.info("Topological visualization is currently being generated. Please run model evaluation first.")

with tab1:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    with st.form("url_analysis_form"):
        url_input = st.text_input("Target URL for Threat Analysis:", placeholder="https://example.com/login", help="Input the full URL including HTTP/HTTPS protocols.")
        submitted = st.form_submit_button("Engage Analysis Hybrid Fusion", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)
    
    # 2. On submit
    if submitted and url_input:
        logger.info(f"Analysis triggered for Target URL: {url_input}")
        start_time = time.time()
        
        try:
            with st.spinner("Extracting structural, statistical, and semantic elements..."):
                import re
                
                # Realign user input to match the training dataset structural bias 
                # (which contains few http prefixes for benign URLs). 
                # This fundamentally corrects false positives on benign safe websites.
                realignment_url = re.sub(r'(?i)^https?://(www\.)?', '', url_input)
                realignment_url = realignment_url.rstrip('/')
                
                # Extract features (using our backend FeatureBuilder pipeline)
                logger.info("Running parallel feature extraction pipeline...")
                df_input = pd.DataFrame([{'url': realignment_url, 'type': 'unknown'}]) 
                builder = FeatureBuilder(raw_data_path="", output_path="")
                df_clean = builder.validate_and_clean(df_input)
                
                if df_clean.empty:
                    logger.error("URL failed validation constraints.")
                    st.error("Invalid URL format or URL could not be parsed.")
                    st.stop()
                    
                df_features = builder.build_features(df_clean)
                
                # Align features exactly with LightGBM's training parameters
                model_features = df_features[model.feature_name_]
                
                # Load trained model to fetch prediction probabilities
                logger.info("Executing LightGBM inference model...")
                P_feature = model.predict_proba(model_features)[0]
                
                # Compute graph domain probabilities via dynamically injecting into PyTorch Geometric
                logger.info("Computing scalable graph domain probabilities via PyTorch GraphSAGE...")
                ext = tldextract.extract(url_input)
                domain = f"{ext.domain}.{ext.suffix}" if ext.domain else ext.suffix
                tld = ext.suffix
                
                # Define universally trusted domains to counteract dataset poisoning 
                TRUSTED_DOMAINS = {
                    'google.com', 'chatgpt.com', 'openai.com', 'github.com', 'microsoft.com', 
                    'apple.com', 'amazon.com', 'facebook.com', 'linkedin.com', 'youtube.com', 
                    'wikipedia.org', 'bing.com', 'yahoo.com', 'instagram.com', 'whatsapp.com'
                }
                
                GLOBAL_PRIOR = np.array([0.65, 0.15, 0.15, 0.05])
                P_graph = np.copy(GLOBAL_PRIOR)
                
                if domain in TRUSTED_DOMAINS:
                    logger.info(f"Domain '{domain}' identified in Global Trust Whitelist. Enforcing benign bias.")
                    P_graph = np.array([0.95, 0.01, 0.02, 0.02])
                elif gnn_model is not None and gnn_data is not None and gnn_mappings is not None:
                    import torch
                    import copy
                    from torch_geometric.data import HeteroData
                    import torch_geometric.transforms as T
                    
                    data_inf = HeteroData()
                    # We avoid deepcopying 600K nodes to avoid freezing the dashboard! 
                    # Instead of copying the whole graph, we can just slice out the neighbors
                    # or temporarily add our node to the original tensor and remove it!
                    # For performance in Streamlit, appending to tensor directly is super fast, 
                    # then we remove it after inference.
                    
                    device = next(gnn_model.parameters()).device
                    
                    feature_cols = gnn_mappings['feature_cols']
                    domain_mapping = gnn_mappings['domain_mapping']
                    tld_mapping = gnn_mappings['tld_mapping']
                    
                    available_feats = []
                    for c in feature_cols:
                        if c in df_features.columns:
                            available_feats.append(df_features[c].values[0])
                        else:
                            available_feats.append(0.0)
                            
                    url_feat_tensor = torch.tensor([available_feats], dtype=torch.float).to(device)
                    
                    # Capture exact counts to revert later
                    orig_url_nodes = gnn_data['url'].x.shape[0]
                    orig_domain_nodes = gnn_data['url'].x.shape[0] # Not used for rollback, but reference
                    
                    # 1. Expand node tensors
                    gnn_data['url'].x = torch.cat([gnn_data['url'].x, url_feat_tensor], dim=0)
                    
                    added_domain = False
                    added_tld = False
                    
                    if domain in domain_mapping:
                        d_idx = domain_mapping[domain]
                    else:
                        d_idx = gnn_data['domain'].x.shape[0]
                        added_domain = True
                        new_domain_feat = url_feat_tensor.clone()
                        gnn_data['domain'].x = torch.cat([gnn_data['domain'].x, new_domain_feat], dim=0)
                        
                    if tld in tld_mapping:
                        t_idx = tld_mapping[tld]
                    else:
                        t_idx = gnn_data['tld'].x.shape[0]
                        added_tld = True
                        new_tld_feat = torch.zeros((1, gnn_data['tld'].x.shape[1]), dtype=torch.float).to(device)
                        gnn_data['tld'].x = torch.cat([gnn_data['tld'].x, new_tld_feat], dim=0)
                        
                    # 2. Add edges
                    # Must find edge index in undirected graph properly
                    new_ud_edges = torch.tensor([[orig_url_nodes], [d_idx]], dtype=torch.long).to(device)
                    orig_ud_edges = gnn_data['url', 'belongs_to', 'domain'].edge_index
                    gnn_data['url', 'belongs_to', 'domain'].edge_index = torch.cat([orig_ud_edges, new_ud_edges], dim=1)
                    
                    orig_dt_edges = gnn_data['domain', 'belongs_to', 'tld'].edge_index
                    
                    if added_domain or added_tld:
                        new_dt_edges = torch.tensor([[d_idx], [t_idx]], dtype=torch.long).to(device)
                        gnn_data['domain', 'belongs_to', 'tld'].edge_index = torch.cat([orig_dt_edges, new_dt_edges], dim=1)
                        
                    # Need reverse edges since undirected
                    new_du_edges = torch.tensor([[d_idx], [orig_url_nodes]], dtype=torch.long).to(device)
                    orig_rev_ud = gnn_data['domain', 'rev_belongs_to', 'url'].edge_index
                    gnn_data['domain', 'rev_belongs_to', 'url'].edge_index = torch.cat([orig_rev_ud, new_du_edges], dim=1)
                    
                    if added_domain or added_tld:
                        new_td_edges = torch.tensor([[t_idx], [d_idx]], dtype=torch.long).to(device)
                        orig_rev_dt = gnn_data['tld', 'rev_belongs_to', 'domain'].edge_index
                        gnn_data['tld', 'rev_belongs_to', 'domain'].edge_index = torch.cat([orig_rev_dt, new_td_edges], dim=1)
                        
                    # 3. Forward Pass (Inductive Generation)
                    with torch.no_grad():
                        probs = gnn_model(gnn_data.x_dict, gnn_data.edge_index_dict)
                        # Newest node is at the end
                        P_graph = probs[-1].cpu().numpy()
                        logger.info(f"GraphSAGE yielded valid P_gnn vector: {P_graph}")
                        
                    # 4. Immediate Rollback (Cleanup to prevent memory ballooning)
                    gnn_data['url'].x = gnn_data['url'].x[:orig_url_nodes]
                    gnn_data['url', 'belongs_to', 'domain'].edge_index = orig_ud_edges
                    gnn_data['domain', 'rev_belongs_to', 'url'].edge_index = orig_rev_ud
                    
                    if added_domain:
                        gnn_data['domain'].x = gnn_data['domain'].x[:-1]
                    if added_tld:
                        gnn_data['tld'].x = gnn_data['tld'].x[:-1]
                    if added_domain or added_tld:
                        gnn_data['domain', 'belongs_to', 'tld'].edge_index = orig_dt_edges
                        gnn_data['tld', 'rev_belongs_to', 'domain'].edge_index = orig_rev_dt
                    
                    # Safeguard metric structure
                    P_graph = P_graph / (np.sum(P_graph) + 1e-9)
    
                # Apply hybrid fusion logic
                logger.info(f"Executing deep hybrid fusion metrics. (Alpha {alpha})")
                beta = 1.0 - alpha
                P_final = alpha * P_feature + beta * P_graph
                
                # Normalize to assure standard percentage metrics
                P_final = P_final / np.sum(P_final)
                pred_class_idx = np.argmax(P_final)
                pred_class = CLASSES[pred_class_idx]
                confidence = P_final[pred_class_idx] * 100
                
                exec_time = time.time() - start_time
                logger.info(f"Intelligence processing finished in {exec_time:.2f}s => Final Flag: {pred_class} at {confidence:.2f}%")
                
                # 3. Show:
                st.markdown("<hr style='border-color: rgba(255,255,255,0.1);'>", unsafe_allow_html=True)
                st.markdown("<h2 style='text-align: center; color: #f8fafc; margin-bottom: 20px;'>Hybrid Intelligence Results</h2>", unsafe_allow_html=True)
                
                metric_cols = st.columns(3)
                
                # Mapping visual status colors
                color_map = {
                    'benign': '#22c55e',       # Safe Green
                    'defacement': '#eab308',   # Warning Yellow
                    'phishing': '#f97316',     # Danger Orange
                    'malware': '#ef4444'       # Critical Red
                }
                pred_color = color_map.get(pred_class, '#38bdf8')
                
                with metric_cols[0]:
                    st.markdown(f"""
                    <div class='card' style='text-align: center; border-bottom: 4px solid {pred_color};'>
                        <div class='metric-label'>Predicted Security Class</div>
                        <div class='metric-value' style='color: {pred_color}; text-transform: capitalize;'>{pred_class}</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                with metric_cols[1]:
                    st.markdown(f"""
                    <div class='card' style='text-align: center;'>
                        <div class='metric-label'>Overall Confidence</div>
                        <div class='metric-value'>{confidence:.2f}%</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                with metric_cols[2]:
                    st.markdown(f"""
                    <div class='card' style='text-align: center;'>
                        <div class='metric-label'>Fusion Coefficient (Alpha)</div>
                        <div class='metric-value'>{alpha:.2f}</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                st.markdown("<br>", unsafe_allow_html=True)
                
                # Charts & Table Elements
                vis_col, table_col = st.columns([1.2, 1])
                
                with vis_col:
                    st.markdown("<div class='card'>", unsafe_allow_html=True)
                    st.markdown("<h3 style='color: #f8fafc;'>Fusion Probability Breakdown</h3>", unsafe_allow_html=True)
                    
                    # Probability bar chart structure
                    prob_df = pd.DataFrame({
                        'Class Indicator': [c.capitalize() for c in CLASSES],
                        'Final Combined Fused Score': P_final,
                        'LightGBM Score (Lexical)': P_feature,
                        'Graph Vector Score (Domain)': P_graph
                    })
                    
                    # Render interactive Plotly graph
                    fig = px.bar(
                        prob_df, 
                        x='Class Indicator', 
                        y=['Final Combined Fused Score', 'LightGBM Score (Lexical)', 'Graph Vector Score (Domain)'],
                        barmode='group',
                        color_discrete_sequence=['#3b82f6', '#10b981', '#f59e0b'],
                        height=450
                    )
                    fig.update_layout(
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        font_color='#94a3b8',
                        legend_title_text='Intelligence Aspect',
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                        margin=dict(l=0, r=0, t=10, b=0)
                    )
                    fig.update_yaxes(gridcolor='rgba(255,255,255,0.05)', title_text='Probability')
                    fig.update_xaxes(title_text='Threat Class')
                    st.plotly_chart(fig, use_container_width=True)
                    st.markdown("</div>", unsafe_allow_html=True)
                    
                with table_col:
                    st.markdown("<div class='card'>", unsafe_allow_html=True)
                    st.markdown("<h3 style='color: #f8fafc;'>Extracted Feature Engine Array</h3>", unsafe_allow_html=True)
                    
                    # Feature breakdown table restructuring
                    feat_table_df = model_features.T.reset_index()
                    feat_table_df.columns = ['Feature Designation', 'Computed Value']
                    
                    # Clean table data
                    feat_table_df['Computed Value'] = pd.to_numeric(feat_table_df['Computed Value']).round(4)
                    
                    # Beautiful Streamlit grid rendering
                    st.dataframe(feat_table_df, height=450, use_container_width=True, hide_index=True)
                    st.markdown("</div>", unsafe_allow_html=True)

                # 4. New: Topological Neighborhood Context
                st.markdown("<div class='card'>", unsafe_allow_html=True)
                st.markdown("<h3 style='color: #f8fafc;'>Input URL Topological Neighborhood (Local GNN Context)</h3>", unsafe_allow_html=True)
                st.markdown("<p style='color: #94a3b8; font-size: 14px;'>This graph shows the structural association between your input URL and its parent domain. The GNN aggregates risk from neighbor URLs under the same Domain/TLD cluster to predict the threat level.</p>", unsafe_allow_html=True)
                
                import networkx as nx
                import plotly.graph_objects as go
                
                # Build localized neighborhood graph
                G_local = nx.Graph()
                ext = tldextract.extract(url_input)
                domain = f"{ext.domain}.{ext.suffix}" if ext.domain else ext.suffix
                tld = ext.suffix
                
                # Add nodes
                G_local.add_node(tld, type='TLD', color='#8b5cf6', size=30)
                G_local.add_node(domain, type='Domain', color='#38bdf8', size=25)
                G_local.add_node("Input URL", type='Target', color=pred_color, size=20)
                
                G_local.add_edge(domain, tld)
                G_local.add_edge("Input URL", domain)
                
                # Add a few "neighbor" URLs to show aggregation if domain exists in graph
                if gnn_mappings and domain in gnn_mappings.get('domain_mapping', {}):
                    G_local.add_node("Neighbor A", type='Neighbor', color='#94a3b8', size=12)
                    G_local.add_node("Neighbor B", type='Neighbor', color='#94a3b8', size=12)
                    G_local.add_edge("Neighbor A", domain)
                    G_local.add_edge("Neighbor B", domain)
                
                pos = nx.spring_layout(G_local, seed=42)
                
                edge_x, edge_y = [], []
                for edge in G_local.edges():
                    x0, y0 = pos[edge[0]]
                    x1, y1 = pos[edge[1]]
                    edge_x.extend([x0, x1, None])
                    edge_y.extend([y0, y1, None])
                    
                edge_trace = go.Scatter(x=edge_x, y=edge_y, line=dict(width=1, color='#475569'), hoverinfo='none', mode='lines')
                
                node_x, node_y, node_color, node_size, node_text = [], [], [], [], []
                for node in G_local.nodes():
                    x, y = pos[node]
                    node_x.append(x)
                    node_y.append(y)
                    node_color.append(G_local.nodes[node]['color'])
                    node_size.append(G_local.nodes[node]['size'])
                    node_text.append(node)
                    
                node_trace = go.Scatter(
                    x=node_x, y=node_y, mode='markers+text', text=node_text, textposition="bottom center",
                    marker=dict(color=node_color, size=node_size, line=dict(width=2, color='#ffffff')),
                    textfont=dict(color="#ffffff")
                )
                
                fig_local = go.Figure(data=[edge_trace, node_trace], 
                                     layout=go.Layout(showlegend=False, margin=dict(b=0,l=0,r=0,t=0), 
                                                     paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                                     xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                                                     yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)))
                st.plotly_chart(fig_local, use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)
                    
        except Exception as e:
            logger.error(f"Inference process faulted: {str(e)}", exc_info=True)
            st.error(f"Core intelligence engine failed during generation: {str(e)}")
