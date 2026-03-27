import os
import tldextract
import networkx as nx
import plotly.graph_objects as go
import numpy as np

def generate_concept_graph(output_html="outputs/reports/domain_graph_concept.html"):
    # 1. Define Sample Data mimicking the dataset
    sample_urls = [
        # Benign Cluster
        ("google.com/search?q=ml", "benign", 0.05),
        ("google.com/mail", "benign", 0.05),
        ("google.com/maps", "benign", 0.05),
        ("drive.google.com/share", "benign", 0.05),
        
        # Phishing Cluster (sharing a domain)
        ("secure-login-paypal.com/update", "phishing", 0.98),
        ("secure-login-paypal.com/verify", "phishing", 0.98),
        ("secure-login-paypal.com/account", "phishing", 0.98),
        
        # Defacement Cluster
        ("hacked-site.org/index.php?id=1", "defacement", 0.85),
        ("hacked-site.org/images/defaced.png", "defacement", 0.85),
        
        # Malware Cluster
        ("free-movie-download.net/player.exe", "malware", 0.99),
        ("free-movie-download.net/codec.zip", "malware", 0.99),
    ]

    G = nx.Graph()

    # Track distinct nodes for multipartite layout
    # Layer 2: URLs, Layer 1: Domains, Layer 0: TLDs
    
    # Add Nodes
    for url, label, risk in sample_urls:
        ext = tldextract.extract(url)
        domain = f"{ext.domain}.{ext.suffix}"
        tld = ext.suffix
        
        # Add TLD Node if not exists
        if not G.has_node(tld):
            G.add_node(tld, type='TLD', layer=0, title=f"TLD: .{tld}<br>Risk Aggregation", color="#8b5cf6", size=40)
            
        # Add Domain Node if not exists
        if not G.has_node(domain):
            # Evaluate risk color based on average URL risks (conceptually)
            is_malicious = "malicious" if risk > 0.5 else "safe"
            color = "#ef4444" if risk > 0.5 else "#22c55e"
            G.add_node(domain, type='Domain', layer=1, 
                       title=f"Domain: {domain}<br>Avg Risk: {risk*100}%<br>Class: {is_malicious}", 
                       color=color, size=30)
            # Edge from Domain to TLD
            G.add_edge(domain, tld)
            
        # Add URL Node
        node_color = "#f97316" if label in ["phishing", "malware", "defacement"] else "#34d399"
        G.add_node(url, type='URL', layer=2, 
                   title=f"URL: {url}<br>Type: {label}", 
                   color=node_color, size=15)
        # Edge from URL to Domain
        G.add_edge(url, domain)

    # Compute Layout (Hierarchical Multi-Partite)
    pos = nx.multipartite_layout(G, subset_key="layer", align="vertical")
    
    # We want a horizontal layout, so let's swap X and Y
    for k, v in pos.items():
        pos[k] = np.array([v[1], -v[0]])

    # Extract Edge Traces
    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=1.5, color='#475569'),
        hoverinfo='none',
        mode='lines')

    # Extract Node Traces
    node_x = []
    node_y = []
    node_color = []
    node_size = []
    node_text = []
    
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_color.append(G.nodes[node]['color'])
        node_size.append(G.nodes[node]['size'])
        node_text.append(G.nodes[node]['title'])

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        hoverinfo='text',
        text=[n for n in G.nodes()],
        textposition="bottom center",
        hovertext=node_text,
        marker=dict(
            showscale=False,
            color=node_color,
            size=node_size,
            line_width=2,
            line_color='#ffffff'
        ))
        
    node_trace.textfont = dict(family="Inter, sans-serif", size=10, color="#cbd5e1")

    # Create Figure
    fig = go.Figure(data=[edge_trace, node_trace],
             layout=go.Layout(
                title=dict(
                    text='<br>Hybrid URL Intelligence: Domain Graph Propagation Concept<br><span style="font-size:14px;color:#94a3b8">URLs (Right) group into Domains (Center) which group into TLDs (Left). Risk scores propogate through the ecosystem.</span>',
                    font=dict(size=20, color="#f8fafc")
                ),
                showlegend=False,
                hovermode='closest',
                margin=dict(b=20,l=5,r=5,t=80),
                paper_bgcolor="#0f172a",
                plot_bgcolor="#0f172a",
                annotations=[ dict(
                    text="This visualization demonstrates how individual URL features are mathematically aggregated into Domain-level<br>and TLD-level probability distributions, allowing the system to identify clusters of phishing campaigns.",
                    showarrow=False,
                    xref="paper", yref="paper",
                    x=0.005, y=-0.002, font=dict(color="#64748b", size=12) ) ],
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                )

    os.makedirs(os.path.dirname(output_html), exist_ok=True)
    fig.write_html(output_html)
    print(f"Graph visualization successfully saved to {output_html}")

if __name__ == "__main__":
    generate_concept_graph()
