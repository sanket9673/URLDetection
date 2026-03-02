# 🛡️ Hybrid Lexical-Graph URL Classifier

A production-ready Enterprise Malicious URL Detection system that combines the power of **LightGBM** (trained on numerical lexical features) with **Domain-Level Graph Intelligence**. The predictions from both models are combined using a tunable weighted hybrid fusion approach to maximize detection accuracy and robustness.

## 🚀 Key Features

* **Advanced Feature Engineering**: Extracts structural, statistical, and suspicious lexical features from raw URLs using optimized, vectorized Pandas operations.
* **Domain-Level Graph Intelligence**: Computes robust domain-level statistical features and graph probabilities to detect previously unseen malicious domains, preventing common data leakage pitfalls.
* **High-Performance Modeling**: Utilizes LightGBM configured for multi-class classification, handling class imbalances via balanced class weights and early stopping.
* **Hybrid Fusion Engine**: Intelligently aggregates LightGBM and Graph model prediction probabilities using a tunable $\alpha$ and $\beta$ weighting formula.
* **Interactive Dashboard**: A beautiful, real-time analytics UI to inspect individual URLs, visualizing probability distributions, feature breakdowns, and final fusion outputs.
* **Extensive Evaluation System**: Robust evaluation framework that auto-generates detailed metrics (Macro F1, Precision, Recall), confusion matrices, ROC curves, and feature importance analyses for both the isolated models and the hybrid system.
* **End-to-End Pipeline**: A fully orchestrated pipeline (`run_pipeline.py`) that runs the entire ML lifecycle with comprehensive logging and error handling.

---

## 📂 Project Structure

```text
Lexical-Graph-URL-Classifier/
├── app/
│   └── dashboard.py               # Interactive UI for real-time URL classification
├── config/
│   └── config.yaml                # Centralized configuration parameters
├── data/                          # Data directory (ignored in git)
│   ├── raw/
│   │   └── malicious_phish.csv    # Raw dataset
│   └── processed/                 # Engineered features & graph data
├── logs/                          # System execution logs 
├── models/                        # Saved LightGBM & Fusion models
├── outputs/                       # Final prediction metrics, plots, and models
├── results/                       # Timestamped experiment tracking & artifacts
├── src/                           # Core source code
│   ├── feature_engineering/       # Lexical feature extraction pipeline
│   ├── graph/                     # Domain-level intelligence computation
│   ├── models/                    # Model training logic (LightGBM)
│   ├── fusion/                    # Hybrid weighting engine
│   └── evaluation/                # Performance analytics and visualizations
├── run_pipeline.py                # Main orchestration script
├── requirements.txt               # Dependencies
└── .gitignore                     # Ignored files & secrets
```

---

## 🛠️ Installation & Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/sanket9673/Lexical-Graph-URL-Classifier.git
   cd Lexical-Graph-URL-Classifier
   ```

2. **Create a virtual environment** (Recommended)
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install the required dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Prepare the Data**
   Place your raw dataset inside the `data/raw/` registry. 
   Expected dataset: `malicious_phish.csv` containing columns for URLs and their corresponding labels.

---

## 🏃‍♂️ Usage

### 1. Run the Full Machine Learning Pipeline
Execute the entire pipeline from end-to-end to process data, extract graph features, train the LightGBM model, tune the fusion weights, and evaluate the system.

```bash
python run_pipeline.py
```

**Pipeline Stages:**
1. **Feature Builder**: Cleans URLs and engineers structural numerical features.
2. **Model Training**: Trains LightGBM and logs performance metrics.
3. **Graph Module**: Generates domain intelligence signatures.
4. **Hybrid Fusion**: Computes optimal weights and evaluates overarching performance.

### 2. Launch the Interactive Dashboard
Spin up the real-time analytics UI to test custom URLs against your newly trained models.

```bash
streamlit run app/dashboard.py
```
*(If the dashboard uses Gradio instead of Streamlit: `python app/dashboard.py`)*

---

## 📊 Evaluation & Logging
The project employs a robust tracking system.
* Every run of the pipeline will log cleanly to `logs/pipeline.log`.
* Each execution automatically generates an isolated experiment directory in `results/experiment_{TIMESTAMP}/` storing:
   * **Metrics**: JSON logs of confusion matrices and detailed F1/Accuracy scores.
   * **Plots**: ROC Curves, Feature Importance visual charts.
   * **Summaries**: Human-readable text reports of the entire experiment.

---

## 🤝 Contributing
Contributions are welcome! Please feel free to open a Pull Request or issue for architectural improvements, bug fixes, or analytical enhancements.

---

## 📝 License
This project is proprietary / open-sourced under the MIT License.
