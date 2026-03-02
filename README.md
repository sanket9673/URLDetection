# 🛡️ Hybrid Lexical-Graph URL Classifier

<p align="center">
  <strong>An Enterprise-Grade Machine Learning System for Malicious URL Detection</strong>
</p>

---

## 📖 Executive Summary
The **Hybrid Lexical-Graph URL Classifier** is a robust, production-ready machine learning pipeline designed to classify URLs into multiple categories (e.g., Benign, Phishing, Malware, Defacement) using a dual-faceted approach. 

Traditional URL classification systems often rely solely on lexical (textual) feature engineering, which struggles against sophisticated obfuscation techniques. This project solves that by combining **LightGBM**—trained on carefully crafted numerical lexical features—with **Domain-Level Graph Intelligence**, which acts as a secondary verification engine looking at the statistical reputation of a given domain. Finally, a mathematical **Hybrid Fusion Engine** aggregates both signals to maximize detection accuracy, precision, and recall.

This project was built to demonstrate end-to-end Machine Learning Engineering capabilities: from advanced feature extraction and handling class imbalance, to preventing data leakage, orchestrating a full ML pipeline, and providing a real-time interactive user interface.

---

## 🧠 Core Architecture & System Design

The system is built upon four primary modules working in tandem:

### 1. Lexical Feature Engineering (`src/feature_engineering/`)
Instead of using raw text embeddings (like TF-IDF or Word2Vec) which can be highly memory-intensive and computationally slow, this engine parses the raw URL structure into mathematically dense numerical features.
* **Structural Features**: URL length, hostname length, path length.
* **Statistical Features**: Count of specific characters (`@`, `?`, `-`, `=`, `.`, `http`, `https`).
* **Suspicious Markers**: Presence of IP addresses (IPv4), shortening services, or unusually high digit-to-letter ratios.
* **Performance**: Utilizes highly optimized, vectorized pandas operations to ensure memory safety and processing speed.

### 2. Domain-Level Graph Intelligence (`src/graph/`)
To combat sophisticated malicious domains that might "look" benign lexically, the graph engine computes intelligence based on historical domain behavior.
* **Mechanism**: Extracts Top-Level Domains (TLDs) and subdomains to track how often a specific domain is associated with malicious behavior.
* **Leakage Prevention**: Strictly computes domain reputation probabilities **only** from the training dataset. This perfectly simulates a real-world production environment to completely prevent data leakage into the validation/test sets.
* **Graph Probabilities**: Generates a smooth, normalized probability distribution indicating the likelihood of a domain belonging to each specific threat class.

### 3. High-Performance Modeling (`src/models/`)
The textual features are fed into a **LightGBM** (Light Gradient Boosting Machine) model.
* **Why LightGBM?**: Chosen for its high efficiency with tabular data, rapid training speed, low memory footprint, and native support for multi-class classification (`multi_logloss`).
* **Class Imbalance**: Utilizes `class_weight='balanced'` alongside stratified train/val/test splitting to ensure minority threat classes (like Malware) are detected with the same rigor as the majority Benign class.

### 4. Hybrid Fusion Engine (`src/fusion/`)
The crowning feature of this architecture is the Decision-Level Fusion Engine. 
* **The Math**: It aggregates LightGBM's lexical probabilities ($P_{lex}$) and the Graph's domain probabilities ($P_{graph}$) using a dynamic weighted average formula:
   $$P_{final} = (\alpha \times P_{lex}) + (\beta \times P_{graph})$$ *(where $\beta = 1 - \alpha$)*.
* **Dynamic Tuning**: The engine auto-tunes the $\alpha$ parameter using the Validation set, selecting the exact optimal balance that yields the highest Macro F1 Score before committing to predictions on the unseen Test set.

---

## 🛠️ Technical Highlights (For the Interviewer)

* **Production-Ready Code Quality**: Written following Software Engineering best practices. Complete with a centralized `config.yaml`, robust exception handling (try-except blocks), absolute paths, and scalable object-oriented/functional design.
* **Comprehensive Logging System**: Print statements are entirely replaced by Python's `logging` module. Execution trails are safely routed to both the console and a central `logs/pipeline.log` file, ensuring traceability.
* **Automated Experiment Tracking**: Every full run automatically creates an isolated, timestamped artifact directory (e.g., `results/experiment_20260302_053610/`). It saves trained models, metadata, JSON metric files, Confusion Matrices, ROC Curves, and Feature Importance charts for reproducible research.
* **Interactive UI (Dashboard)**: A sleek real-time application (`app/dashboard.py`) built to immediately validate the model against new, unseen URLs.

---

## 📂 Repository Structure

```text
Lexical-Graph-URL-Classifier/
├── app/
│   └── dashboard.py               # Streamlit application for real-time URL inference
├── config/
│   └── config.yaml                # Centralized hyperparameters and pipeline configuration
├── data/                          # Data directory (ignored in git)
│   ├── raw/                       # Expected raw input (e.g., malicious_phish.csv)
│   └── processed/                 # Output of the feature engineering & graph pipelines
├── logs/                          # System execution logs (pipeline.log)
├── models/                        # Serialized LightGBM & Fusion artifacts (.pkl)
├── outputs/                       # Final visual outputs & system metrics
├── results/                       # Automated, timestamped experiment tracking logs
├── src/                           # Core Machine Learning codebase
│   ├── feature_engineering/       # Lexical rules & vectorized feature builder
│   ├── graph/                     # Domain statistics & probability intelligence
│   ├── models/                    # LightGBM training, validation, and serialization
│   ├── fusion/                    # Hyperparameter tuning for decision fusion
│   ├── evaluation/                # Analytics engine (ROC, Feature Importance, Confusion Matrix)
│   ├── logger_config.py           # Central logging initialization
│   └── utils.py                   # Helper functions (OS pathing, saving artifacts)
├── run_pipeline.py                # The central orchestrator for the ML lifecycle
└── requirements.txt               # Required Python packages
```

---

## � Setup & Installation

**1. Clone the repository**
```bash
git clone https://github.com/sanket9673/Lexical-Graph-URL-Classifier.git
cd Lexical-Graph-URL-Classifier
```

**2. Setup Virtual Environment**
```bash
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

**3. Install Dependencies**
```bash
pip install -r requirements.txt
```

**4. Data Preparation**
Place the raw dataset inside the `data/raw/` registry. 
*Expected format:* `malicious_phish.csv` containing columns for `url` and `type` (the label).

---

## 🏃‍♂️ Executing the ML Pipeline

To trigger the end-to-end Machine Learning pipeline—which includes feature building, graph computation, model training, hyperparameter tuning, model fusion, and experiment evaluations—simply run the orchestrator script:

```bash
python run_pipeline.py
```

When this script finishes, it will print a final summarized report to the console detailing the **Macro F1 Scores** across the isolated LightGBM model versus the Hybrid Fusion system, definitively quantifying the improvement.

---

## 📊 Evaluation Metrics & Visualizations
Upon a successful pipeline run, navigate to the `results/experiment_{TIMESTAMP}/` directory to inspect the generated analytics:
- **`plots/lightgbm_roc.png`**: ROC curves detailing True Positive vs False Positive rates for each individual threat class.
- **`confusion_matrices/lightgbm_confusion.png`**: Heatmap exposing exactly where the model struggles (e.g., misclassifying Malware as Phishing).
- **`feature_importance/top_features.csv`**: A ranked list indicating which engineered features (e.g., `url_length`, `count_@`) provided the most information gain to the LightGBM trees.

---

## ⚡ Launching the Interactive Dashboard
To interact with the trained models and test live URLs, launch the dashboard:

```bash
streamlit run app/dashboard.py
```
This UI provides immediate feedback, including Final Predicted Class, Confidence Percentages, and a transparency breakdown of the computed Lexical Features.

---

## 🤝 Contact & Contributions
Designed and deployed by [Sanket Chavhan](https://github.com/sanket9673). 
Open to contributions and feedback regarding the architecture or model performance.
