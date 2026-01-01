# interpretability.py
import shap
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

class Explainer:
    """
    A wrapper for SHAP explainability for tree-based models (Random Forest, XGBoost, etc.)
    """

    def __init__(self, model, X_train: pd.DataFrame, X_test: pd.DataFrame):
        """
        Initialize the explainer.
        Args:
            model: Fitted tree-based model
            X_train: Training features (used if needed for background data)
            X_test: Test features for computing SHAP values
        """
        self.model = model
        self.X_train = X_train
        self.X_test = X_test
        self.explainer = shap.TreeExplainer(model)
        self.shap_values_sample = None

    def compute_shap_sample(self, n_samples: int = 1000, random_state: int = 42):
        """
        Compute SHAP values for a representative sample of the test set.
        """
        X_sample = self.X_test.sample(n_samples, random_state=random_state)
        shap_values = self.explainer(X_sample)

        # Handle multi-class output
        if len(shap_values.shape) == 3:
            self.shap_values_sample = shap_values[:, :, 1]  # focus on class 1
        else:
            self.shap_values_sample = shap_values

        # return self.shap_values_sample

    def plot_global_summary(self):
        """
        Generate a global SHAP summary (beeswarm) plot.
        """
        if self.shap_values_sample is None:
            self.compute_shap_sample()
        shap.plots.beeswarm(self.shap_values_sample)

    def plot_waterfall(self, idx: int, title: str):
        """
        Generate a waterfall plot for a single observation.
        Args:
            idx: Index in X_test
            title: Plot title
        """
        row = self.X_test.loc[[idx]]
        sv = self.explainer(row)

        # Handle multi-class output
        if len(sv.shape) == 3:
            exp = shap.Explanation(
                values=sv.values[0, :, 1],
                base_values=sv.base_values[0, 1],
                data=row.iloc[0],
                feature_names=self.X_test.columns.tolist()
            )
        else:
            exp = shap.Explanation(
                values=sv.values[0],
                base_values=sv.base_values[0],
                data=row.iloc[0],
                feature_names=self.X_test.columns.tolist()
            )

        plt.figure(figsize=(5, 4))  
        plt.title(title, fontsize=12)
        shap.plots.waterfall(exp, show=True)
        plt.tight_layout()

    def plot_dependence(self, feature: str):
        """
        Generate a SHAP dependence plot for a given feature using the sampled SHAP values.
        Args:
            feature: Name of the feature
        """
        if self.shap_values_sample is None:
            self.compute_shap_sample()

        # Extract feature column index
        if feature not in self.X_test.columns:
            raise ValueError(f"Feature '{feature}' not in X_test columns")

        shap.plots.scatter(self.shap_values_sample[:, feature])
