# Phishing URL Detection

This project is a Machine Learning-based Web Application designed to classify URLs as legitimate or phishing. The application extracts 30 different features from a given URL to predict its safety using a Gradient Boosting Classifier, ultimately giving the user a percentage score of the URL's safety.

## Table of Contents
- [Project Overview](#project-overview)
- [Directory Structure](#directory-structure)
- [Model Details](#model-details)
- [Feature Engineering](#feature-engineering)
- [Tech Stack](#tech-stack)
- [Installation and Setup](#installation-and-setup)
- [Usage](#usage)
- [Deployment](#deployment)

## Project Overview
Malicious links and phishing attacks are major cybersecurity threats. This application provides users with an easy-to-use interface to verify whether a URL is a phishing attempt. 
It uses a Pre-trained Machine Learning Model pipeline that evaluates lexical strings, domain intelligence, and web traffic metadata without executing the malicious requests on the user's end.

## Directory Structure

```plaintext
URLDetection/
│
├── app.py                   # Main Flask Application
├── feature.py               # Feature Extraction Module (Extracts 30 Features)
├── requirements.txt         # Project Dependencies
├── Procfile                 # Deployment configurations for production
├── phishing.csv             # The original dataset used to train the model
├── Method-1-phising URL detection_project.ipynb # Jupyter Notebook containing EDA and Model Training
│
├── pickle/
│   └── model.pkl            # Trained ML Model (Gradient Boosting Classifier)
│
├── templates/               # Contains HTML templates for the Web Application
│   ├── index.html           # Home page
│   ├── about.html           # About page
│   └── detection.html       # Result page showing whether a URL is safe or phishing
│
└── static/                  # Contains static CSS/JS/image assets
```

## Model Details
The system utilizes a Gradient Boosting algorithm. The trained model (`model.pkl`) is loaded into memory when the Flask server starts. 
- **Input:** A 30-dimensional array of numerical features (derived from the URL string and external request data).
- **Output:** A continuous probability score of the given URL being legitimate.
- **Classification Categories:**
  - `-1` (Phishing/Unsafe)
  - `1` (Legitimate/Safe)
  - `0` (Suspicious)

## Feature Engineering
The success of this machine learning system heavily relies on `feature.py`. The module computes the following 30 diverse characteristics for any given URL:

1. **Address Bar Based Features:** Using IP, Long URLs, Short URL Services, "@" Symbol presence, Double Slash Redirecting, Prefix-Suffix analysis, Number of Subdomains, HTTPS protocol compliance, Domain Registration Length, Favicon status, Standard Port usage, and HTTPS in Domain URL.
2. **Abnormal Based Features:** Request URL (percentage of external objects), Anchor URL (percentage of external/suspicious anchor tags), Links in `<Script>` and `<Link>` tags, Server Form Handler (SFH) checks, Info Email strings, and General Abnormal URLs.
3. **HTML and JavaScript Based Features:** Website Forwarding trace, Status Bar Customizations, Disable Right Click, pop-up windows alerts, and Iframe Redirections.
4. **Domain Based Features:** Age of Domain, DNS Recording existence, Website Traffic (Alexa Rank), PageRank insights, Google Indexing, Links pointing to the page, and advanced Statistical Reports matching blacklist IPs/Domains.

## Tech Stack
- **Backend Framework:** Flask 2.0.2
- **Machine Learning:** Scikit-Learn (Gradient Boosting Trees)
- **Data Manipulation:** NumPy, Pandas
- **Web Scraping & Parsing:** BeautifulSoup4, Requests
- **Domain Utilities:** whois, googlesearch-python, python-dateutil
- **Production Server:** Gunicorn

## Installation and Setup

### 1. Clone the repository
Navigate to your desired directory and clone/navigate to this codebase.

### 2. Create a virtual environment (Optional but Recommended)
```bash
python -m venv venv
source venv/bin/activate
# On Windows use: venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Application
Execute the following command to start the Flask application:
```bash
python app.py
```
The application will be accessible at: `http://localhost:5000/`

## Usage
1. Provide the suspected custom URL in the dashboard's text input box on the home page.
2. Click the check/detect button.
3. The `FeatureExtraction` class dynamically evaluates the URL over live HTTP/WHOIS requests to build a 30-feature vector.
4. The system routes the numerical features into the classifier and outputs an actionable "safe percentage" so you know if it is reliable to proceed.

## Deployment
The repository includes a `Procfile` configured for easy deployment on Platform-as-a-Service providers (e.g., Heroku, Render).
```text
web: gunicorn app:app
```
To deploy:
- Ensure the PaaS provider is connected to your version control repository or CLI.
- The platform will automatically install Python instances, read `requirements.txt`, and start the Gunicorn webserver mapped to `app.py`.
