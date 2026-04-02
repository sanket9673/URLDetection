# GraphSAGE Integration & Heterogeneous Graph Architecture

## Architectural Changes
The URL Classification Engine has been substantially upgraded. We deprecated the naive statistical domain reputation calculation logic (`domain_graph.py`) and successfully introduced a sophisticated **Heterogeneous GraphSAGE model** (via PyTorch Geometric). 

This architectural change shifts the paradigm from mathematical historical tracking down to structural graph representation learning.

### Node and Edge Abstractions
The URL internet topology is inherently multipartite. We successfully represented this as a bipartite, heterogeneous graph comprising three node structures:
1. **URL Nodes**: Encoded with 40 deep lexical features extracted during URL character analyses.
2. **Domain Nodes**: Meaningful continuous distributions representing behavioral averages of interconnected URLs sharing the same registered domain.
3. **TLD Nodes**: One-hot encoded embeddings identifying distinct global suffixes.

The bipartite edges are defined strictly hierarchically:
- `URL -> Domain`
- `Domain -> TLD`

## Inductive Bias and Unseen Threat Generalization
Statistical models inherently fail on *Zero-Day Domains* (domains never seen during training). If an attacker purchases a new domain and registers it yesterday, historical statistical probability models would map it to 0.0 feature importance, essentially "guessing" blindly.

**GraphSAGE**, conversely, operates under a robust **Inductive Bias**. It does not learn an embedding *per node map* but rather learns *an aggregator function* that generalizes how to pool features from a node's local neighborhood.

### How it seamlessly resolves zero-day domains (Runtime Inference Dynamics):
When an unseen URL is queried into the `dashboard.py` runtime architecture:
1. We extract the unanalyzed URL's 40 lexical features to instantly produce a URL Node tensor.
2. `tldextract` isolates the Domain. If the domain is *unseen*, GraphSAGE doesn't panic. Instead, we dynamically inject a transient **Domain Node** into the in-memory Heterogeneous Graph.
3. We project dynamic structural connectivity (`URL -> Domain -> TLD`), ensuring it connects to existing global entities (like the '.xyz' TLD node).
4. The learned $SAGEConv$ aggregation equation pulls spatial data backward through these fresh connections, projecting an immediate algorithmic embedding for the zero-day threat purely via inductive reasoning on its position alongside similar nodes and its own deep lexical footprint.

## Fusion Synergy ( α * P_lexical + β * P_gnn )
Instead of returning explicit class integers, GraphSAGE explicitly outputs a 4-dimensional continuous probability space (`P_gnn`). This vector represents its confidence matrix over (Benign, Defacement, Phishing, Malware).

The secondary stage in `hybrid_fusion.py` mathematically isolates this output and blends it continuously alongside the independent LightGBM confidence intervals (`P_lexical`). This synergistic ensemble allows optimal convergence, dynamically shifting weight (Alpha) toward whichever spatial topology provides a sharper resolution of the threat.
