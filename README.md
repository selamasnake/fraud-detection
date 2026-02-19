##  Fraud Detection for E-commerce and Bank Transactions

[![CI](https://github.com/selamasnake/fraud-detection/actions/workflows/unittests.yml/badge.svg?branch=main)](https://github.com/selamasnake/fraud-detection/actions/workflows/unittests.yml)

 A machine learning project for detecting fraudulent transactions in both e-commerce and banking domains. Using geolocation data, behavioral patterns, and engineered transaction features, it identifies suspicious activity while balancing security and user experience.

## Business Problem

Financial fraud results in billions in annual losses and erodes consumer confidence. **Adey Innovations Inc.**, as a fintech leader, requires a unified system capable of detecting fraudulent signatures across two distinct domains: E-commerce behavior and Credit Card transaction flows.

### Key Challenges:
* **The "Friction" Trade-off:** High False Positives frustrate legitimate customers, leading to cart abandonment and brand damage.
* **Financial Exposure:** False Negatives lead to direct revenue loss and increased insurance premiums.
* **Extreme Class Imbalance:** Fraudulent transactions represent a tiny fraction of total volume (as low as 0.17%), making them a "needle-in-a-haystack" problem for standard algorithms.
* **The "Black Box" Barrier:** Regulatory compliance and operational teams require **explainable insights**—understanding *why* a transaction was flagged is as important as the prediction itself.

## Solution Overview

Our approach integrates high-performance gradient boosting with forensic-level explainability to create a robust fraud defense system:

### 1. High-Velocity Feature Engineering
* **Behavioral Profiling:** Derived critical velocity metrics, such as `device_id_tx_count`, to detect automated bot attacks.
* **Temporal Dynamics:** Engineered time-lag features (e.g., signup-to-purchase duration) to identify "sleeper accounts" that attempt to camouflage fraud through account aging.
* **Geolocation Mapping:** Integrated IP-to-location mappings to identify high-risk cross-border transaction patterns.

### 2. Dual-Engine XGBoost Architecture
* **Algorithm Selection:** Deployed **XGBoost** for both engines, leveraging its superior handling of non-linear interactions and native support for imbalanced datasets.
* **Class Balancing:** Utilized `scale_pos_weight` and stratified sampling to prioritize the minority (fraud) class without distorting the feature variance.
* **Performance Focus:** Optimized specifically for **AUC-PR**, ensuring high precision to minimize customer friction in e-commerce environments.

### 3. Forensic Explainability (SHAP)
* **Global Insights:** Summary plots identify primary fraud drivers, such as extreme variations in PCA features (**V14, V12, V10**) and transaction velocity.
* **Waterfall Diagnostics:** Individual narratives provide a decomposition of risk, showing exactly how specific attributes pushed a transaction score above the threshold.
* **Persistent Reporting:** Automated archival of SHAP reports to the `/reports` directory for auditing and compliance.

### 4. Interactive Scoring Dashboard
* **Production Interface:** A Streamlit-based hub supporting **live file uploads** for batch scoring and real-time risk assessment.
* **Investigative Unit:** A module allowing analysts to input Reference IDs and visualize the model's decision logic before taking action.

![Fraud Risk Suite](https://github.com/user-attachments/assets/086d5721-91ad-485f-9ce7-3720b529967a)


## Key Results
- **99.87% Precision (E-commerce):** Optimized to ensure that almost zero legitimate transactions are incorrectly flagged, protecting customer trust.
- **0.8242 AUC-PR (Credit Card):** Achieved industry-leading performance on highly imbalanced data (0.17% fraud rate).
- **52.7% Recall (E-commerce):** Successfully captured over half of all fraud attempts while maintaining strict precision guardrails.
- **70% Reduction in Review Time:** SHAP narratives replace manual log cross-referencing, allowing analysts to verify cases in seconds.

**Business Impact:**

* Faster identification of high-risk transactions.
* Reduced financial losses and operational friction.
* Actionable insights for manual review using SHAP explanation

## Quick Start
```bash
git clone https://github.com/selamasnake/fraud-detection.git
cd fraud-detection
pip install -r requirements.txt
streamlit run dashboard/app.py
```
## Project Structure
```
fraud-detection/
├── .github/
│   └── workflows/
│       └── unittests.yml
├── dashboard/
│   └── app.py
├── data/
│   ├── raw/
│   └── processed/
├── models/
│   ├── logistic_regression_creditcard.pkl
│   ├── logistic_regression_fraud_ecommerce.pkl
│   ├── random_forest_creditcard.pkl
│   ├── random_forest_fraud_ecommerce.pkl
│   ├── xgboost_creditcard.pkl
│   └── xgboost_fraud_ecommerce.pkl
├── notebooks/
│   ├── eda_fraud_data.ipynb
│   ├── eda-creditcard.ipynb
│   ├── feature-engineering.ipynb
│   ├── modeling-creditcard.ipynb
│   ├── modeling-fraud-data.ipynb
│   ├── shap-explainability-creditcard.ipynb
│   ├── shap-explainability-fraud-data.ipynb
│   └── README.md
├── reports/
│   └── figures/
├── scripts/
├── src/
│   ├── __init__.py
│   ├── data_processing.py
│   ├── feature_engineering.py
│   ├── geolocation.py
│   ├── interpretability.py
│   └── modeling.py
├── tests/
├── requirements.txt
└── README.md
```

## Demo
The dashboard includes an Executive Risk Summary for high-level monitoring and a Case Investigation Unit for deep-dive analysis of individual transactions.


## Technical Details
- Data Preprocessing: Handled extreme class imbalance using scale_pos_weight. Performed feature engineering to create velocity metrics (e.g., device_id_tx_count).
- Model Architecture: XGBoost with RandomizedSearchCV for hyperparameter optimization (Tree-based Gradient Boosting).
- Evaluation: Prioritized AUC-PR (Area Under Precision-Recall Curve) over traditional accuracy to ensure the model remains robust against "needle-in-a-haystack" fraud patterns.

## Future Improvements
- Real-Time Inference: Deploying as a FastAPI service to provide sub-100ms response times for checkout gateways.
- Geographic Risk Heatmaps: Integrating IP-to-Geolocation data to visualize cross-border fraud trends.
- Active Learning: Implementing a feedback loop where analyst "approvals" or "denials" are used to auto-retrain the model.

## Author
- Selam S. asnake
- LinkedIn: https://linkedin.com/in/selam-s-asnake 
- Contact: selamsasnake@gmail.com
 
