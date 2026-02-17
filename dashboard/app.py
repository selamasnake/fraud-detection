import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
import plotly.express as px
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Any, Optional, Dict

# --- CONSTANTS ---
SHAP_SAMPLE_SIZE: int = 100
RANDOM_SEED: int = 42
FRAUD_THRESHOLD: float = 0.5
CHART_COLOR_FRAUD: str = "#e74c3c"
CHART_COLOR_LEGIT: str = "#2ecc71"
UI_ACCENT_COLOR: str = "#98f1e5"
SAMPLE_SCATTER_SIZE: int = 1000
MAX_DISPLAY_IDS: int = 100
IMPORTANCE_TOP_N: int = 10  # Restricted to top 10 as requested

# --- PATH SETUP ---
BASE_DIR: Path = Path(__file__).resolve().parent.parent
SRC_PATH: str = str(BASE_DIR / "src")
if SRC_PATH not in sys.path:
    sys.path.append(SRC_PATH)

from interpretablility import Explainer

# --- CONFIGURATION OBJECTS ---
@dataclass(frozen=True)
class DatasetConfig:
    name: str
    model_file: str
    data_file: str
    target: str
    id_col: str
    drop_cols: List[str]
    precision: str
    recall: str
    auc_pr: str
    rationale: str
    directives: List[str]

ECOMM_CONFIG = DatasetConfig(
    name="E-commerce Fraud",
    model_file="xgboost_fraud_ecommerce.pkl",
    data_file="fraud_data_features.csv",
    target="class",
    id_col="user_id",
    drop_cols=["class", "purchase_time", "signup_time", "user_id", "device_id", "ip_address"],
    precision="99.87%",
    recall="52.7%",
    auc_pr="0.7103",
    rationale="XGBoost selected for highest AUC-PR (0.7103). High precision minimizes customer friction, while lower recall suggests a need for behavioral guardrails against sleeper accounts.",
    directives=[
        "Monitor 'Sleeper Accounts': High account age can camouflage fraud signatures.",
        "Implement 'Change-of-Pattern' alerts for sudden velocity spikes on aged accounts.",
        "Auto-decline transactions where device_id_tx_count exceeds 95th percentile risk."
    ]
)

CREDIT_CONFIG = DatasetConfig(
    name="Credit Card Fraud",
    model_file="xgboost_creditcard.pkl",
    data_file="creditcard_cleaned.csv",
    target="Class",
    id_col="index",
    drop_cols=["Class"],
    precision="96.05%",
    recall="76.84%",
    auc_pr="0.8242",
    rationale="XGBoost selected for industry-leading AUC-PR (0.8242). Captures non-linear interactions in PCA features while maintaining low false-positive rates.",
    directives=[
        "Prioritize manual review for extreme variations in V14, V12, and V10.",
        "Mandate MFA for transactions exhibiting high-variance PCA signatures.",
        "Dynamic thresholding required for V4 and V11 to adapt to evolving patterns."
    ]
)

# --- UTILITIES ---
def get_model_components(model: Any) -> Tuple[Any, Optional[Any]]:
    if hasattr(model, 'steps'):
        scaler = next((s for n, s in model.steps if hasattr(s, "transform") and n != 'model'), None)
        classifier = model.steps[-1][1]
        return classifier, scaler
    return model, None

def prepare_shap_data(model: Any, X_raw: pd.DataFrame, bg_raw: pd.DataFrame, feature_cols: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    classifier, scaler = get_model_components(model)
    if scaler:
        X_trans = pd.DataFrame(scaler.transform(X_raw), columns=feature_cols)
        bg_trans = pd.DataFrame(scaler.transform(bg_raw), columns=feature_cols)
        return X_trans, bg_trans
    return X_raw, bg_raw

def calculate_friction_risk(precision_str: str) -> float:
    return 100.0 - float(precision_str.strip('%'))

@st.cache_resource
def load_data_suite(config: DatasetConfig) -> Tuple[Any, pd.DataFrame, List[str]]:
    model = joblib.load(BASE_DIR / "models" / config.model_file)
    df = pd.read_csv(BASE_DIR / "data" / "processed" / config.data_file)
    if config.id_col == "index" and "index" not in df.columns:
        df = df.reset_index()
    feature_cols = list(model.feature_names_in_) if hasattr(model, 'feature_names_in_') else \
                   [c for c in df.columns if c not in config.drop_cols and c != config.id_col]
    return model, df, feature_cols

# --- MAIN UI EXECUTION ---
st.set_page_config(page_title="Adey Innovations | Fraud Risk Suite", layout="wide")

dataset_map: Dict[str, DatasetConfig] = {ECOMM_CONFIG.name: ECOMM_CONFIG, CREDIT_CONFIG.name: CREDIT_CONFIG}
dataset_choice: str = st.sidebar.selectbox("Active Dataset", list(dataset_map.keys()))
active_cfg: DatasetConfig = dataset_map[dataset_choice]

model, df, feature_cols = load_data_suite(active_cfg)
menu: str = st.sidebar.radio("Navigation", ["Executive Risk Summary", "Case Investigation Unit"])

if menu == "Executive Risk Summary":
    st.title(f"Fraud Risk Management: {active_cfg.name}")
    
    # ROW 1: Metrics
    st.subheader("1. Operational Performance & Model Integrity")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Transaction Volume", f"{len(df):,}")
    m2.metric("System Integrity (AUC-PR)", active_cfg.auc_pr)
    m3.metric("Detection Recall", active_cfg.recall)
    
    friction = calculate_friction_risk(active_cfg.precision)
    m4.metric("Customer Friction Risk", f"{friction:.2f}%", delta="Low Friction", delta_color="normal")
    st.info(f"**Model Rationale:** {active_cfg.rationale}")
    st.divider()

    # ROW 2: Technical Importance (VERTICAL PLOT - TOP 10)
    st.subheader("2. Model Architecture Importance (Information Gain)")
    classifier, _ = get_model_components(model)
    try:
        importance = pd.Series(classifier.get_booster().get_score(importance_type='gain')).sort_values(ascending=False).head(IMPORTANCE_TOP_N)
        fig_imp = px.bar(
            x=importance.index, 
            y=importance.values, 
            labels={'x': 'Model Feature', 'y': 'Gain Score (Top 10)'},
            color_discrete_sequence=[UI_ACCENT_COLOR]
        )
        st.plotly_chart(fig_imp, use_container_width=True)
    except:
        st.info("Feature importance data is restricted for this model type.")
    st.divider()

    # ROW 3: Financial Exposure
    st.subheader("3. Financial Exposure Distribution")
    val_col: str = 'purchase_value' if 'purchase_value' in df.columns else 'Amount'
    fig_dist = px.histogram(df, x=val_col, color=active_cfg.target, barmode='overlay',
                            marginal="box", color_discrete_map={0: CHART_COLOR_LEGIT, 1: CHART_COLOR_FRAUD})
    st.plotly_chart(fig_dist, use_container_width=True)
    st.divider()

    # ROW 4: Variance Analysis
    st.subheader("4. Feature Variance Analysis & Composition")
    c1, c2 = st.columns([2, 1])
    with c1:
        x_anom: str = 'V14' if 'V14' in df.columns else 'age'
        y_anom: str = 'V12' if 'V12' in df.columns else 'purchase_value'
        st.write(f"Cluster Variance: {x_anom} vs {y_anom}")
        fig_scatter = px.scatter(df.sample(min(SAMPLE_SCATTER_SIZE, len(df))), x=x_anom, y=y_anom, color=active_cfg.target,
                                 color_continuous_scale=[CHART_COLOR_LEGIT, CHART_COLOR_FRAUD], opacity=0.6)
        st.plotly_chart(fig_scatter, use_container_width=True)
    with c2:
        st.write("Class Ratio")
        pie_data = df[active_cfg.target].value_counts().reset_index()
        pie_data.columns = ['Status', 'Count']
        pie_data['Status'] = pie_data['Status'].map({0: 'Legitimate', 1: 'Fraud'})
        fig_pie = px.pie(pie_data, values='Count', names='Status', hole=0.5,
                         color_discrete_sequence=[CHART_COLOR_LEGIT, CHART_COLOR_FRAUD])
        st.plotly_chart(fig_pie, use_container_width=True)
    st.divider()
     
    # ROW 5: Global SHAP
    st.subheader("5. Behavioral Feature Impact Analysis (SHAP)")
    if st.button("Generate Global Behavioral Report"):
        with st.spinner("Analyzing global risk patterns..."):
            bg_raw = df[feature_cols].sample(SHAP_SAMPLE_SIZE, random_state=RANDOM_SEED)
            X_trans, bg_trans = prepare_shap_data(model, bg_raw, bg_raw, feature_cols)
            classifier, _ = get_model_components(model)
            explainer_obj = Explainer(classifier, bg_trans, X_trans)
            shap_values = explainer_obj.explainer(X_trans)
            fig, ax = plt.subplots(figsize=(12, 6))
            shap.summary_plot(shap_values, X_trans, show=False)
            st.pyplot(fig)
    st.divider()

    # ROW 6: Directives
    st.subheader("6. Operational Directives")
    for directive in active_cfg.directives:
        st.write(f"- {directive}")

else:
    # --- PAGE 2: CASE INVESTIGATION UNIT ---
    st.title("Case Investigation Unit")
    investigation_mode: str = st.radio("Queue Filter", ["Full Registry", "High-Risk Identified"], horizontal=True)
    display_df: pd.DataFrame = df[df[active_cfg.target] == 1] if "High-Risk" in investigation_mode else df

    selected_id: Any = st.selectbox(f"Reference ID ({active_cfg.id_col})", display_df[active_cfg.id_col].unique()[:MAX_DISPLAY_IDS])
    record: pd.DataFrame = df[df[active_cfg.id_col] == selected_id]
    X_raw: pd.DataFrame = record[feature_cols]

    risk_score: float = float(model.predict_proba(X_raw)[0][1])
    is_fraud: int = int(record[active_cfg.target].values[0])

    st.subheader("Transaction Summary")
    k1, k2, k3 = st.columns(3)
    k1.metric("Predicted Probability", f"{risk_score:.2%}")
    k2.metric("Audit Status", "Confirmed Fraud" if is_fraud == 1 else "Legitimate")
    
    if risk_score > FRAUD_THRESHOLD:
        if is_fraud == 1: st.success("Verdict: True Positive")
        else: st.warning("Verdict: False Positive")
    
    st.divider()
    st.subheader("Local Decision Logic (Waterfall)")
    if st.button("Generate Narrative"):
        with st.spinner("Decomposing prediction..."):
            bg_raw = df[feature_cols].sample(SHAP_SAMPLE_SIZE, random_state=RANDOM_SEED)
            X_trans, bg_trans = prepare_shap_data(model, X_raw, bg_raw, feature_cols)
            classifier, _ = get_model_components(model)
            explainer_obj = Explainer(classifier, bg_trans, X_trans)
            shap_vals = explainer_obj.explainer(X_trans)[0]
            shap_vals.data = X_raw.iloc[0].values 
            fig, ax = plt.subplots(figsize=(10, 6))
            shap.plots.waterfall(shap_vals, show=False)
            st.pyplot(fig)