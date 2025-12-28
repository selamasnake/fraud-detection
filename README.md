##  fraud-detection 
This project develops machine learning models to detect fraudulent transactions in both e-commerce and banking domains. By leveraging geolocation data, behavioral analytics, and engineered transaction features, it identifies suspicious activity while carefully balancing security and user experience.

To address the extreme class imbalance inherent in fraud datasets, the project incorporates advanced resampling techniques like SMOTE. Model interpretability is a key focus, with explainability tools such as SHAP enabling transparent, actionable insights.

The overall goal is to provide accurate, reliable, and explainable fraud detection, helping businesses reduce financial losses, improve customer trust, and maintain operational efficiency in real-time transaction monitoring.

#### Project Structure
- `data/` — contains raw and processed datasets (ignored in git).

- `notebooks/` — Jupyter notebooks for analysis and development:

    - `eda-fraud-data.ipynb` — initial exploration of e-commerce data.

    - `eda-creditcard.ipynb `— analysis of bank transaction data.

    - `feature_engineering.ipynb` — time-based features, velocity metrics, and SMOTE implementation.

    - `modeling-fraud-data.ipynb`: Model training, evaluation, and selection for fraud data

    - `modeling-creditcard.ipynb`: Model training, evaluation, and selection for credit card data

- `src/` — Python modules for core logic:

    - `data_processing.py` — data loading and cleaning helpers.

    - `feature_engineering.py` — implementation of the FeatureEngineer class.

    - `modeling.py` - model building and training module

- .github/workflows/unittests.yml — CI workflow for automated testing.

#### How to Use

1. Clone the repo and install dependencies from `requirements.txt`.  
2. Use the notebooks to explore data, engineer features, and handle class imbalance. 
3. Processed datasets in `data/processed/` are ready for model training.

#### Requirements
See requirements.txt for the list of libraries, including pandas, scikit-learn, imblearn, and matplotlib.