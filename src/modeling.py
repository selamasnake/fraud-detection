import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import clone
import joblib

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

class Modeling:
    """End-to-end class for training and evaluating fraud detection models."""

    def __init__(self):
        """Initialize model registry and train-test placeholders."""
        self.models = {}
        self.results = []
        self.X_train, self.X_test, self.y_train, self.y_test = [None] * 4

    def prepare_data(self, X, y, test_size=0.2):
        """Perform a stratified train-test split to preserve class imbalance."""
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            stratify=y,
            random_state=42
        )
        #return self.X_train, self.X_test, self.y_train, self.y_test

    def train_logistic_baseline(self):
        """Train and evaluate an interpretable Logistic Regression baseline model."""
        pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('model', LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                random_state=42
            ))
        ])
        pipe.fit(self.X_train, self.y_train)
        self.models["Logistic Regression"] = pipe

        result = self._evaluate(pipe, "Logistic Regression")
        self.results.append(result)

        return pd.DataFrame([result])


    def train_random_forest(self, param_grid=None):
        """Train and evaluate a Random Forest model with optional hyperparameter tuning."""
        rf = RandomForestClassifier(
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        )

        pipe = Pipeline([
            ('model', rf)
        ])

        if param_grid:
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            grid = GridSearchCV(
                pipe,
                param_grid=param_grid,
                scoring="average_precision",
                cv=cv,
                n_jobs=-1
            )
            grid.fit(self.X_train, self.y_train)
            model = grid.best_estimator_
        else:
            model = pipe.fit(self.X_train, self.y_train)

        self.models["Random Forest"] = model

        result = self._evaluate(model, "Random Forest")
        self.results.append(result)
        return pd.DataFrame([result])

    def _evaluate(self, model, name):
        """Evaluate a model using threshold-optimized F1 and fraud-focused metrics."""
        probs = model.predict_proba(self.X_test)[:, 1]
        threshold = self._find_best_threshold(probs)
        preds = (probs >= threshold).astype(int)

        return {
            "Model": name,
            "AUC-PR": average_precision_score(self.y_test, probs),
            "F1": f1_score(self.y_test, preds),
            "Precision": precision_score(self.y_test, preds),
            "Recall": recall_score(self.y_test, preds),
            "Best Threshold": round(threshold, 3),
            "Confusion Matrix": confusion_matrix(self.y_test, preds)
        }

    def _find_best_threshold(self, probs):
        """Determine the probability threshold that maximizes the F1-score."""
        precision, recall, thresholds = precision_recall_curve(self.y_test, probs)
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-9)
        return thresholds[np.argmax(f1_scores[:-1])]

    def cross_validated_auc_pr(self, model_key, X, y, n_splits=5):
        """
        Compute mean and standard deviation of AUC-PR using stratified K-fold CV,
        safely using an unfitted clone of the model to avoid data leakage.
        """
        # Get the pipeline for the requested model
        model_pipeline = self.models.get(model_key)
        if model_pipeline is None:
            raise ValueError(f"No model found with key '{model_key}'")
        
        # Create fresh, unfitted copy
        model_clone = clone(model_pipeline)

        # Stratified K-Fold
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

        # Compute cross-validated AUC-PR
        scores = cross_val_score(
            model_clone,
            X,
            y,
            scoring="average_precision",
            cv=skf,
            n_jobs=-1
        )
        return scores.mean(), scores.std()

    def compare_models(self):
        """Return a sorted DataFrame comparing all trained models."""
        return pd.DataFrame(self.results).sort_values(
            by="AUC-PR",
            ascending=False
        ).reset_index(drop=True)
    def save_model(self, model_key, path):
        """Save a trained model pipeline to disk."""
        if model_key not in self.models:
            raise ValueError(f"No model found with key '{model_key}'")
        joblib.dump(self.models[model_key], path)

