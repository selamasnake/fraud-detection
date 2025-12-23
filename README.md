##  fraud-detection 
This project develops  machine learning models to detect fraud in e-commerce and banking transactions. By integrating geolocation data and engineering behavioral features, it identifies suspicious patterns while balancing the trade-off between security and user experience. To handle extreme class imbalance, the project utilizes SMOTE resampling and explainability tools like SHAP to provide accurate, transparent, and actionable fraud insights.

#### Project Structure
- `data/` — contains raw and processed datasets (ignored in git).

- `notebooks/` — Jupyter notebooks for analysis and development:

    - `eda-fraud-data.ipynb` — initial exploration of e-commerce data.

    - `eda-creditcard.ipynb `— analysis of bank transaction data.

    - `feature_engineering.ipynb` — time-based features, velocity metrics, and SMOTE implementation.

- `src/` — Python modules for core logic:

    - `data_processing.py` — data loading and cleaning helpers.

    - `feature_engineering.py` — implementation of the FeatureEngineer class.

- .github/workflows/unittests.yml — CI workflow for automated testing.

#### How to Use

1. Clone the repo and install dependencies from `requirements.txt`.  
2. Use the notebooks to explore data, engineer features, and handle class imbalance. 
3. Processed datasets in `data/processed/` are ready for model training.

#### Requirements
See requirements.txt for the list of libraries, including pandas, scikit-learn, imblearn, and matplotlib.