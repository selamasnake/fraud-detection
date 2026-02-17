import logging
from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import clone
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    GridSearchCV,
    cross_val_score
)
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    precision_recall_curve
)
import xgboost as xgb
import joblib

# -------------------------
# Logging
# -------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -------------------------
# Constants
# -------------------------
EPSILON = 1e-9
DEFAULT_METRIC = "average_precision"

# -------------------------
# Config Dataclass
# -------------------------
@dataclass
class ModelConfig:
    """Configuration for modeling, train-test split, and hyperparameters."""
    test_size: float = 0.2
    random_state: int = 42
    n_splits: int = 5
    max_iter_logreg: int = 1000
    class_weight: str = "balanced"
    ensemble_n_jobs: int = -1



# -------------------------
# Modeling Class
# -------------------------
class Modeling:
    """End-to-end class for training, evaluating, comparing, and selecting fraud detection models."""

    def __init__(self, config: Optional[ModelConfig] = None):
        """
        Initialize the model registry, results list, train/test data placeholders, and configuration.

        Args:
            config (Optional[ModelConfig]): Custom configuration object. Defaults to ModelConfig().
        """
        self.models: Dict[str, Any] = {}
        self.results: list[Dict[str, Any]] = []
        self.X_train: Optional[pd.DataFrame] = None
        self.X_test: Optional[pd.DataFrame] = None
        self.y_train: Optional[pd.Series] = None
        self.y_test: Optional[pd.Series] = None
        self.config = config or ModelConfig()

    # -------------------------
    # Data Preparation
    # -------------------------
    def prepare_data(self, X: pd.DataFrame, y: pd.Series) -> None:
        """
        Perform a stratified train-test split to preserve class imbalance.

        Args:
            X (pd.DataFrame): Feature matrix.
            y (pd.Series): Target variable.
        """
        logger.info("Performing stratified train-test split...")
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X,
            y,
            test_size=self.config.test_size,
            stratify=y,
            random_state=self.config.random_state
        )

    # -------------------------
    # Model Training
    # -------------------------
    def train_logistic_baseline(self) -> pd.DataFrame:
        """
        Train and evaluate a Logistic Regression baseline model with StandardScaler.

        Returns:
            pd.DataFrame: Single-row DataFrame containing evaluation metrics for the trained model.
        """
        logger.info("Training Logistic Regression baseline...")
        pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('model', LogisticRegression(
                max_iter=self.config.max_iter_logreg,
                class_weight=self.config.class_weight,
                random_state=self.config.random_state
            ))
        ])
        pipe.fit(self.X_train, self.y_train)
        self.models["Logistic Regression"] = pipe

        result = self._evaluate(pipe, "Logistic Regression")
        self.results.append(result)
        return pd.DataFrame([result])

    def train_random_forest(self, param_grid: Optional[dict] = None) -> pd.DataFrame:
        """
        Train and evaluate a Random Forest classifier with optional hyperparameter tuning via GridSearchCV.

        Args:
            param_grid (Optional[dict]): Dictionary of hyperparameters for GridSearchCV. Example: 
                                         {"model__n_estimators": [100, 200], "model__max_depth": [5, 10]}.

        Returns:
            pd.DataFrame: Single-row DataFrame containing evaluation metrics for the trained model.
        """
        logger.info("Training Random Forest...")
        rf = RandomForestClassifier(
            class_weight=self.config.class_weight,
            random_state=self.config.random_state,
            n_jobs=self.config.ensemble_n_jobs
        )
        pipe = Pipeline([('model', rf)])

        if param_grid:
            cv = StratifiedKFold(
                n_splits=self.config.n_splits,
                shuffle=True,
                random_state=self.config.random_state
            )
            grid = GridSearchCV(
                pipe,
                param_grid=param_grid,
                scoring=DEFAULT_METRIC,
                cv=cv,
                n_jobs=self.config.ensemble_n_jobs
            )
            grid.fit(self.X_train, self.y_train)
            model = grid.best_estimator_
        else:
            model = pipe.fit(self.X_train, self.y_train)

        self.models["Random Forest"] = model
        result = self._evaluate(model, "Random Forest")
        self.results.append(result)
        return pd.DataFrame([result])
    
    def train_xgboost(self, params: Optional[dict] = None) -> pd.DataFrame:
        """
        Train and evaluate an XGBoost classifier.

        Args:
            params (Optional[dict]): Hyperparameters to pass to XGBClassifier. Example: 
                                     {"max_depth": 5, "n_estimators": 200}.

        Returns:
            pd.DataFrame: Single-row DataFrame containing evaluation metrics for the trained model.
        """
        clf = xgb.XGBClassifier(
            eval_metric="logloss",
            random_state=self.config.random_state,
            **(params or {})
        )
        pipe = Pipeline([('model', clf)])
        pipe.fit(self.X_train, self.y_train)

        self.models["XGBoost"] = pipe
        result = self._evaluate(pipe, "XGBoost")
        self.results.append(result)
        return pd.DataFrame([result])

    # -------------------------
    # Model Evaluation
    # -------------------------
    def _evaluate(self, model: Pipeline, name: str) -> Dict[str, Any]:
        """
        Evaluate a model using threshold-optimized F1-score and fraud metrics.

        Args:
            model (Pipeline): Trained sklearn pipeline or estimator.
            name (str): Name of the model for logging and results.

        Returns:
            Dict[str, Any]: Dictionary containing AUC-PR, F1, Precision, Recall, Best Threshold, and Confusion Matrix.
        """
        probs = model.predict_proba(self.X_test)[:, 1]
        threshold = self._find_best_threshold(probs)
        preds = (probs >= threshold).astype(int)

        logger.info(f"Evaluated {name} | Threshold: {threshold:.3f}")

        return {
            "Model": name,
            "AUC-PR": average_precision_score(self.y_test, probs),
            "F1": f1_score(self.y_test, preds),
            "Precision": precision_score(self.y_test, preds),
            "Recall": recall_score(self.y_test, preds),
            "Best Threshold": round(threshold, 3),
            "Confusion Matrix": confusion_matrix(self.y_test, preds)
        }

    def _find_best_threshold(self, probs: np.ndarray) -> float:
        """
        Determine the probability threshold that maximizes the F1-score.

        Args:
            probs (np.ndarray): Predicted probabilities for the positive class.

        Returns:
            float: Optimal threshold maximizing F1-score.
        """
        precision, recall, thresholds = precision_recall_curve(self.y_test, probs)
        f1_scores = 2 * (precision * recall) / (precision + recall + EPSILON)
        return thresholds[np.argmax(f1_scores[:-1])]

    def cross_validated_auc_pr(
        self, model_key: str, X: pd.DataFrame, y: pd.Series
    ) -> Tuple[float, float]:
        """
        Compute mean and standard deviation of AUC-PR using stratified K-fold cross-validation.

        Args:
            model_key (str): Key of the trained model in self.models.
            X (pd.DataFrame): Feature matrix.
            y (pd.Series): Target variable.

        Returns:
            Tuple[float, float]: Mean and standard deviation of cross-validated AUC-PR.

        Raises:
            ValueError: If the model_key does not exist in self.models.
        """
        model_pipeline = self.models.get(model_key)
        if model_pipeline is None:
            raise ValueError(f"No model found with key '{model_key}'")
        model_clone = clone(model_pipeline)
        skf = StratifiedKFold(
            n_splits=self.config.n_splits,
            shuffle=True,
            random_state=self.config.random_state
        )
        scores = cross_val_score(
            model_clone, X, y, scoring=DEFAULT_METRIC, cv=skf, n_jobs=self.config.ensemble_n_jobs
        )
        return scores.mean(), scores.std()

    # -----------------------
    # Model Comparison & Selection
    # -----------------------
    def compare_models(self) -> pd.DataFrame:
        """
        Compare all trained models sorted by AUC-PR in descending order.

        Returns:
            pd.DataFrame: DataFrame of evaluation metrics for all trained models.
        """
        return pd.DataFrame(self.results).sort_values(by="AUC-PR", ascending=False).reset_index(drop=True)

    def select_best_model(self) -> Tuple[str, dict]:
        """
        Select the best model based on highest AUC-PR.

        Prints a concise rationale including AUC-PR and F1-score.

        Returns:
            Tuple[str, dict]: Model name and corresponding evaluation metrics.

        Raises:
            ValueError: If no models have been trained yet.
        """
        if not self.results:
            raise ValueError("No models have been trained yet.")

        df = self.compare_models()
        best_row = df.iloc[0]
        model_name = best_row["Model"]
        rationale = (
            f"Selected '{model_name}' as best model due to highest AUC-PR "
            f"({best_row['AUC-PR']:.3f}) and competitive F1 ({best_row['F1']:.3f})."
        )
        print(rationale)
        return model_name, best_row.to_dict()

    def save_model(self, model_key: str, path: str):
        """
        Save a trained model pipeline to disk.

        Args:
            model_key (str): Key of the trained model in self.models.
            path (str): File path to save the model.

        Raises:
            ValueError: If the model_key does not exist in self.models.
        """
        if model_key not in self.models:
            raise ValueError(f"No model found with key '{model_key}'")
        joblib.dump(self.models[model_key], path)