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
        
    graph_df = None
    if os.path.exists(GRAPH_PATH):
        graph_df = pd.read_parquet(GRAPH_PATH)
    else:
        logger.warning(f"Graph features missing at {GRAPH_PATH}. Falling back to default probabilities.")
        
    alpha = 0.5
    if os.path.exists(METRICS_PATH):
        try:
            with open(METRICS_PATH, "r") as f:
                metrics = json.load(f)
                alpha = metrics.get('best_alpha', 0.5)
                logger.info(f"Loaded tuned alpha factor: {alpha}")
        except Exception as e:
            logger.warning(f"Could not load best_alpha, defaulting to 0.5. Error: {e}")
            
    return model, graph_df, alpha

try:
    model, graph_df, alpha = load_assets()
    logger.info("Intelligence engine ready online.")
except Exception as e:
    st.error(f"Intelligence Engine offline: {str(e)}")
    st.stop()

# 1. URL input field
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
            
            # Compute graph domain probabilities via extraction matching
            logger.info("Computing scalable graph domain probabilities...")
            ext = tldextract.extract(url_input)
            domain = f"{ext.domain}.{ext.suffix}" if ext.domain else ext.suffix
            tld = ext.suffix
            
            # Define universally trusted domains to counteract dataset poisoning 
            # (e.g. users uploading malware to Google Drive flags 'google.com' as malware in the dataset)
            TRUSTED_DOMAINS = {
                'google.com', 'chatgpt.com', 'openai.com', 'github.com', 'microsoft.com', 
                'apple.com', 'amazon.com', 'facebook.com', 'linkedin.com', 'youtube.com', 
                'wikipedia.org', 'bing.com', 'yahoo.com', 'instagram.com', 'whatsapp.com'
            }
            
            # Global dataset priors (approx: 65% benign, 15% defacement, 15% phishing, 5% malware)
            GLOBAL_PRIOR = np.array([0.65, 0.15, 0.15, 0.05])
            
            # Default fallback
            P_graph = np.copy(GLOBAL_PRIOR)
            
            if domain in TRUSTED_DOMAINS:
                logger.info(f"Domain '{domain}' identified in Global Trust Whitelist. Enforcing benign bias.")
                P_graph = np.array([0.95, 0.01, 0.02, 0.02])
            elif graph_df is not None:
                graph_cols = [f'domain_class_{c}_prob' for c in range(4)]
                tld_cols = [f'tld_class_{c}_prob' for c in range(4)]
                
                domain_match = graph_df[graph_df['registered_domain'] == domain]
                if not domain_match.empty:
                    freq = domain_match['domain_frequency'].iloc[0]
                    # If frequency is very low, it's noisy, so we blend it with global prior
                    raw_graph = domain_match[graph_cols].iloc[0].values
                    if freq < 5:
                        P_graph = (raw_graph + GLOBAL_PRIOR) / 2.0
                        logger.info(f"Found specific node graph projection for domain: {domain} (Low freq: {freq}, smoothed)")
                    else:
                        P_graph = raw_graph
                        logger.info(f"Found specific node graph projection for domain: {domain} (Freq: {freq})")
                else:
                    tld_match = graph_df[graph_df['tld'] == tld]
                    if not tld_match.empty:
                        raw_tld = tld_match[tld_cols].iloc[0].values
                        # Blend TLD probability with Global Prior to prevent harsh penalization (e.g., heavily penalizing all .coms)
                        P_graph = (raw_tld * 0.1) + (GLOBAL_PRIOR * 0.9)
                        logger.info(f"Node graph domain missing, utilizing smoothed TLD fallback projection: {tld}")
                
                # Safeguard normalization
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
                
    except Exception as e:
        logger.error(f"Inference process faulted: {str(e)}", exc_info=True)
        st.error(f"Core intelligence engine failed during generation: {str(e)}")
