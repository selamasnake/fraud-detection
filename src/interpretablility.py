import shap
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional

@dataclass
class SHAPConfig:
    """
    Configuration for SHAP explainability.
    """
    n_samples: int = 1000
    random_state: int = 42
    global_plot_figsize: Tuple[int, int] = (6, 4)
    waterfall_figsize: Tuple[int, int] = (6, 4)
    dependence_plot_figsize: Tuple[int, int] = (8, 5)
    target_class_index: int = 1  # Used for binary/multi-class models

class Explainer:
    """
    A wrapper for SHAP explainability for tree-based models (Random Forest, XGBoost, etc.)
    """

    def __init__(self, model, X_train: pd.DataFrame, X_test: pd.DataFrame, config: Optional[SHAPConfig] = None):
        """
        Initialize the explainer.

        Args:
            model: Fitted tree-based model
            X_train: Training features (used for background data)
            X_test: Test features for computing SHAP values
            config: Optional SHAP configuration dataclass
        """
        self.model = model
        self.X_train = X_train
        self.X_test = X_test
        self.config = config or SHAPConfig()
        self.explainer = shap.TreeExplainer(model)
        self.shap_values_sample = None

    def _extract_shap_class(self, shap_values) -> shap.Explanation:
        """
        Extract SHAP values for the target class if multi-class.

        Args:
            shap_values: SHAP output from explainer

        Returns:
            SHAP values for target class or original values for binary model
        """
        if len(shap_values.shape) == 3:  # Multi-class
            return shap_values[:, :, self.config.target_class_index]
        return shap_values

    def compute_shap_sample(self) -> None:
        """
        Compute SHAP values for a representative sample of the test set.
        """
        X_sample = self.X_test.sample(self.config.n_samples, random_state=self.config.random_state)
        shap_values = self.explainer(X_sample)
        self.shap_values_sample = self._extract_shap_class(shap_values)

    def plot_global_summary(self) -> None:
        """
        Generate a global SHAP summary (beeswarm) plot.
        """
        if self.shap_values_sample is None:
            self.compute_shap_sample()
        plt.figure(figsize=self.config.global_plot_figsize)
        shap.plots.beeswarm(self.shap_values_sample)
        plt.tight_layout()
        plt.show()

    def plot_waterfall(self, idx: int, title: str) -> None:
        """
        Generate a waterfall plot for a single observation.

        Args:
            idx: Index in X_test
            title: Plot title
        """
        if idx not in self.X_test.index:
            raise ValueError(f"Index {idx} not found in X_test")

        row = self.X_test.loc[[idx]]
        sv = self.explainer(row)
        sv_class = self._extract_shap_class(sv)

        if len(sv_class.shape) == 3:  # Multi-class
            exp = shap.Explanation(
                values=sv_class.values[0, :, self.config.target_class_index],
                base_values=sv_class.base_values[0, self.config.target_class_index],
                data=row.iloc[0],
                feature_names=self.X_test.columns.tolist()
            )
        else:
            exp = shap.Explanation(
                values=sv_class.values[0],
                base_values=sv_class.base_values[0],
                data=row.iloc[0],
                feature_names=self.X_test.columns.tolist()
            )

        plt.figure(figsize=self.config.waterfall_figsize)
        plt.title(title, fontsize=12)
        shap.plots.waterfall(exp, show=True)
        plt.tight_layout()
        plt.show()

    def plot_dependence(self, feature: str) -> None:
        """
        Generate a SHAP dependence plot for a given feature using the sampled SHAP values.

        Args:
            feature: Name of the feature
        """
        if self.shap_values_sample is None:
            self.compute_shap_sample()

        if feature not in self.X_test.columns:
            raise ValueError(f"Feature '{feature}' not in X_test columns")

        plt.figure(figsize=self.config.dependence_plot_figsize)
        shap.plots.scatter(self.shap_values_sample[:, feature])
        plt.tight_layout()
        plt.show()

    def _map_values_to_features(self, df: pd.DataFrame, value_col: str, features: List[str]) -> List[float]:
        """
        Map a column of values to a given list of features, returning 0 if missing.

        Args:
            df: DataFrame with 'feature' column and value column
            value_col: Column containing values to map
            features: List of features to extract values for

        Returns:
            List of values corresponding to features
        """
        value_map = df.set_index("feature")[value_col].to_dict()
        return [value_map.get(f, 0.0) for f in features]

    def get_builtin_importance(self) -> pd.DataFrame:
        """
        Returns built-in model feature importance as a sorted DataFrame.

        Returns:
            DataFrame with columns ['feature', 'importance'] sorted descending
        """
        if not hasattr(self.model, "feature_importances_"):
            raise AttributeError("Model does not have 'feature_importances_' attribute")
        
        return (
            pd.DataFrame({
                "feature": self.X_test.columns,
                "importance": self.model.feature_importances_
            })
            .sort_values(by="importance", ascending=False)
            .reset_index(drop=True)
        )

    def get_shap_importance(self) -> pd.DataFrame:
        """
        Returns mean absolute SHAP importance per feature.

        Returns:
            DataFrame with columns ['feature', 'mean_abs_shap'] sorted descending
        """
        if self.shap_values_sample is None:
            self.compute_shap_sample()
        
        return (
            pd.DataFrame({
                "feature": self.X_test.columns,
                "mean_abs_shap": np.abs(self.shap_values_sample.values).mean(axis=0)
            })
            .sort_values(by="mean_abs_shap", ascending=False)
            .reset_index(drop=True)
        )

    def compare_importance(
        self,
        top_n: Optional[int] = None
    ) -> Tuple[pd.DataFrame, List[str], List[float], List[float]]:
        """
        Compare top-N built-in importance vs SHAP importance.

        Args:
            top_n: Number of top features to consider (defaults to 5)

        Returns:
            comparison_df: DataFrame with columns ['Built-in Importance', 'SHAP Importance', 'Overlap']
            features: Ordered list of unique features for plotting
            builtin_vals: List of built-in importance values for features
            shap_vals: List of SHAP mean absolute values for features
        """
        top_n = top_n or 5  # Could be linked to self.config if desired

        builtin_df = self.get_builtin_importance()
        shap_df = self.get_shap_importance()

        top_builtin = builtin_df.head(top_n)["feature"].tolist()
        top_shap = shap_df.head(top_n)["feature"].tolist()

        # Comparison table with overlap
        comparison_df = pd.DataFrame({
            "Built-in Importance": top_builtin,
            "SHAP Importance": top_shap
        })
        overlap = set(top_builtin) & set(top_shap)
        comparison_df["Overlap"] = comparison_df.apply(
            lambda row: "Yes" if (
                row["Built-in Importance"] in overlap or row["SHAP Importance"] in overlap
            ) else "",
            axis=1
        )

        # Prepare data for plotting
        features = list(dict.fromkeys(top_builtin + top_shap))  # preserves order
        builtin_vals = self._map_values_to_features(builtin_df, "importance", features)
        shap_vals = self._map_values_to_features(shap_df, "mean_abs_shap", features)

        return comparison_df, features, builtin_vals, shap_vals