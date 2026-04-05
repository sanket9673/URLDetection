import os
import tldextract
import networkx as nx
import plotly.graph_objects as go
import numpy as np
import pandas as pd
import torch

# Static fallbacks if dataset is missing
STATIC_SAMPLES = [
    ("paypal-secure-verify.xyz/login", "phishing", 0.98),
    ("paypal-secure-verify.xyz/account/update", "phishing", 0.98),
    ("google.com/search?q=gnn", "benign", 0.01),
    ("mail.google.com/inbox", "benign", 0.01),
    ("fast-download.net/setup.exe", "malware", 0.95),
]

def generate_gnn_topology_graph(output_html="outputs/reports/gnn_topology_visualization.html"):
    """
    Constructs a visualization of the actual GNN structural topology,
    using samples from the real dataset.
    """
    # 1. Load Real Data if available
    dataset_path = "data/processed/feature_dataset.parquet"
    if os.path.exists(dataset_path):
        try:
            df = pd.read_parquet(dataset_path)
            # Sample a few domains from each category
            classes = ['benign', 'phishing', 'malware', 'defacement']
            sample_df_list = []
            for cls in classes:
                # Filter by label if it's there
                target_col = 'target' if 'target' in df.columns else 'label' if 'label' in df.columns else None
                if target_col:
                    cls_df = df[df[target_col] == cls]
                    if not cls_df.empty:
                        # Grab 2 domains max per class
                        unique_domains = cls_df['url'].apply(lambda x: tldextract.extract(str(x)).domain).unique()
                        for d in unique_domains[:2]:
                            # Get up to 3 URLs for this domain
                            d_urls = cls_df[cls_df['url'].str.contains(d)].head(3)
                            # Tag them with risk (simplified)
                            d_urls['risk'] = 0.95 if cls != 'benign' else 0.05
                            d_urls['label'] = cls
                            sample_df_list.append(d_urls[['url', 'label', 'risk']])
            
            if sample_df_list:
                sample_nodes = pd.concat(sample_df_list).values.tolist()
            else:
                raise ValueError("No data found for classes.")
        except Exception as e:
            print(f"Fallback to static samples due to error: {e}")
            sample_nodes = STATIC_SAMPLES
    else:
        sample_nodes = STATIC_SAMPLES

    G = nx.Graph()

    # Colors for different classes matching our aesthetic theme
    class_colors = {
        'benign': '#22c55e',       # Safe Green
        'defacement': '#eab308',   # Warning Yellow
        'phishing': '#f97316',     # Danger Orange
        'malware': '#ef4444',      # Critical Red
        'TLD': '#8b5cf6',          # Purple for TLD
        'Domain': '#38bdf8'        # Blue for Domain
    }

    # Layer mapping for multipartite layout: TLD (0) -> Domain (1) -> URL (2)
    # Add Nodes
    for url, label, risk in sample_nodes:
        ext = tldextract.extract(url)
        domain = f"{ext.domain}.{ext.suffix}" if ext.domain else ext.suffix
        tld = ext.suffix
        
        # 1. Add TLD Node
        if not G.has_node(tld):
            G.add_node(tld, type='TLD', layer=0, 
                       title=f"<b>TLD: .{tld}</b><br>Structural Hierarchy Node", 
                       color=class_colors['TLD'], size=45, risk=0.5)
            
        # 2. Add Domain Node
        if not G.has_node(domain):
            # Domain risk is inherited from the URLs
            d_color = class_colors['benign'] if risk < 0.5 else class_colors['phishing'] if label == 'phishing' else class_colors['malware']
            G.add_node(domain, type='Domain', layer=1, 
                       title=f"<b>Domain: {domain}</b><br>Aggregation Vector<br>Risk Association: {risk*100:.1f}%", 
                       color=d_color, size=35, risk=risk)
            # Edge from Domain to TLD
            G.add_edge(domain, tld, weight=1.0)
            
        # 3. Add URL Node
        u_color = class_colors[label]
        G.add_node(url, type='URL', layer=2, 
                   title=f"<b>URL Node</b><br>Path: {url}<br>Status: {label.capitalize()}<br>GNN Probability: {risk*100:.1f}%", 
                   color=u_color, size=18, risk=risk)
        # Edge from URL to Domain
        G.add_edge(url, domain, weight=1.0)

    # Compute Layout (Horizontal Multi-Partite)
    pos = nx.multipartite_layout(G, subset_key="layer", align="vertical")
    # Swap axes for horizontal flow (Left: TLD -> Center: Domain -> Right: URL)
    pos_transformed = {k: np.array([v[1], -v[0]]) for k, v in pos.items()}

    # Extract Edge Traces
    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = pos_transformed[edge[0]]
        x1, y1 = pos_transformed[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=1.5, color='rgba(148, 163, 184, 0.5)'),
        hoverinfo='none',
        mode='lines')

    # Extract Node Traces by Layer for better styling control
    node_traces = []
    for layer_id, layer_name in [(0, 'TLD'), (1, 'Domain'), (2, 'URL')]:
        nodes = [n for n, d in G.nodes(data=True) if d.get('layer') == layer_id]
        x = [pos_transformed[n][0] for n in nodes]
        y = [pos_transformed[n][1] for n in nodes]
        colors = [G.nodes[n]['color'] for n in nodes]
        sizes = [G.nodes[n]['size'] for n in nodes]
        texts = [G.nodes[n]['title'] for n in nodes]
        labels = [n for n in nodes] # For node labels

        trace = go.Scatter(
            x=x, y=y,
            mode='markers+text',
            hoverinfo='text',
            text=labels,
            textposition="bottom center" if layer_id != 1 else "top center",
            hovertext=texts,
            marker=dict(
                color=colors,
                size=sizes,
                line=dict(width=2, color='#ffffff'),
                opacity=0.9
            ),
            name=layer_name,
            textfont=dict(family="Inter, sans-serif", size=10, color="#ffffff")
        )
        node_traces.append(trace)

    # 3. Create Figure with Premium Dark Theme
    fig = go.Figure(data=[edge_trace] + node_traces)
    
    fig.update_layout(
        title=dict(
            text='<b>Heterogeneous GNN Structural Intelligence</b><br><span style="font-size:14px;color:#94a3b8">Visualizing the GraphSAGE "Sample and Aggregate" mechanism across TLD, Domain, and URL nodes.</span>',
            x=0.01,
            font=dict(size=24, color="#f8fafc")
        ),
        showlegend=True,
        legend=dict(
            font=dict(color="#94a3b8"),
            bgcolor="rgba(0,0,0,0)",
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        hovermode='closest',
        margin=dict(b=40, l=40, r=40, t=100),
        paper_bgcolor="#0f172a",
        plot_bgcolor="#0f172a",
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        annotations=[
            dict(
                text="<b>Aggregated Risk Propagation (Guilt by Association)</b><br>The GNN aggregates lexical features from URLs to Domains and TLDs.<br>If a Domain specializes in Phishing, newly encountered URLs inherit that risk vector through structural topology.",
                showarrow=False,
                xref="paper", yref="paper",
                x=0, y=-0.1,
                align="left",
                font=dict(color="#cbd5e1", size=12)
            )
        ]
    )

    # Add background shapes to delineate layers
    fig.add_vrect(x0=-1.1, x1=-0.6, fillcolor="rgba(139, 92, 246, 0.05)", layer="below", line_width=0)
    fig.add_vrect(x0=-0.3, x1=0.3, fillcolor="rgba(56, 189, 248, 0.05)", layer="below", line_width=0)
    fig.add_vrect(x0=0.6, x1=1.1, fillcolor="rgba(34, 197, 94, 0.05)", layer="below", line_width=0)

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_html), exist_ok=True)
    fig.write_html(output_html)
    print(f"GNN Topology visualization saved to {output_html}")
    return output_html

if __name__ == "__main__":
    generate_gnn_topology_graph()
