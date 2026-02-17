import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Any, Optional

# --- CONSTANTS ---
BACKGROUND_SAMPLE_SIZE = 100
RANDOM_SEED = 42
DEFAULT_SELECTION_LIMIT = 100
FRAUD_THRESHOLD = 0.5

# --- PATH SETUP ---
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR / "src") not in sys.path:
    sys.path.append(str(BASE_DIR / "src"))

from interpretablility import Explainer

@dataclass
class DatasetConfig:
    name: str
    model_filename: str
    data_filename: str
    drop_columns: List[str]
    id_column: str

ECOMM_CONFIG = DatasetConfig(
    name="E-commerce Fraud",
    model_filename="xgboost_fraud_ecommerce.pkl",
    data_filename="fraud_data_features.csv",
    drop_columns=["class", "purchase_time", "signup_time", "user_id", "device_id", "ip_address"],
    id_column="user_id",
)

CREDIT_CONFIG = DatasetConfig(
    name="Credit Card Fraud",
    model_filename="xgboost_creditcard.pkl",
    data_filename="creditcard_cleaned.csv",
    drop_columns=["Class"],
    id_column="index",
)

def get_classifier(model: Any) -> Any:
    """Extracts the core booster/classifier from a scikit-learn pipeline or object."""
    return model.named_steps['model'] if hasattr(model, 'named_steps') else model

def transform_data(model: Any, x_input: pd.DataFrame, bg_samples: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Applies scaling transformations if a scaler exists in the model pipeline."""
    if hasattr(model, 'named_steps') and 'scaler' in model.named_steps:
        scaler = model.named_steps['scaler']
        return scaler.transform(x_input), scaler.transform(bg_samples)
    return x_input, bg_samples

@st.cache_resource
def load_assets(config: DatasetConfig) -> Tuple[Any, pd.DataFrame, List[str]]:
    model = joblib.load(BASE_DIR / "models" / config.model_filename)
    df = pd.read_csv(BASE_DIR / "data" / "processed" / config.data_filename)
    
    if config.id_column == "index":
        df = df.reset_index()
        
    feature_cols = [c for c in df.columns if c not in config.drop_columns and c != config.id_column]
    return model, df, feature_cols

# --- UI EXECUTION ---
st.set_page_config(page_title="Fraud Intelligence", layout="wide")

selected_name = st.sidebar.selectbox("Data Source", [ECOMM_CONFIG.name, CREDIT_CONFIG.name])
active_config = ECOMM_CONFIG if selected_name == ECOMM_CONFIG.name else CREDIT_CONFIG

model, df, feature_cols = load_assets(active_config)
classifier = get_classifier(model)

st.title(f"Fraud Analysis: {active_config.name}")

st.subheader("Individual Case Investigation")

selected_id = st.selectbox(
    f"Search by {active_config.id_column.replace('_', ' ').title()}", 
    df[active_config.id_column].unique()[:DEFAULT_SELECTION_LIMIT]
)

record = df[df[active_config.id_column] == selected_id]
X_raw = record[feature_cols]

risk_score = model.predict_proba(X_raw)[0][1]
status = "Fraudulent" if risk_score > FRAUD_THRESHOLD else "Legitimate"
color_style = "inverse" if risk_score > FRAUD_THRESHOLD else "normal"

st.metric("Model Risk Score", f"{risk_score:.2%}", delta=status, delta_color=color_style)

st.write("### Model Logic Explanation")
if st.button("Run Explainer"):
    with st.spinner("Calculating SHAP values..."):
        bg_raw = df[feature_cols].sample(BACKGROUND_SAMPLE_SIZE, random_state=RANDOM_SEED)
        X_trans, bg_trans = transform_data(model, X_raw, bg_raw)

        explainer_obj = Explainer(classifier, bg_trans, X_trans)
        shap_vals = explainer_obj.explainer(X_trans)[0]
        shap_vals.data = X_raw.iloc[0].values 
        
        fig, ax = plt.subplots(figsize=(10, 6))
        shap.plots.waterfall(shap_vals, show=False)
        plt.title(f"Feature Contribution (ID: {selected_id})")
        st.pyplot(fig)