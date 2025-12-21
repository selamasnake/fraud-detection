import os
import pandas as pd
import numpy as np
from scipy.stats import entropy


class DataLoader:
    """Load and save CSV datasets."""

    def __init__(self, path: str | None = None):
        self.path = path
        self.df = None

    def load_data(self, path: str | None = None) -> pd.DataFrame:
        file_path = path or self.path
        if not file_path:
            raise ValueError("No file path specified.")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        self.df = pd.read_csv(file_path, low_memory=False)
        return self.df

    def save_data(self, output_path: str) -> None:
        if self.df is None:
            raise ValueError("No data loaded.")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        self.df.to_csv(output_path, index=False)


class DataCleaner:
    """Clean data: handles data overview, missing values, duplicates, datetime."""

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()

    def data_overview(self) -> dict:
        """Returns dataset information and dtypes."""
        return self.df.info()

    def missing_values_summary(self) -> pd.DataFrame:
        """Summarize missing values per column."""
        missing = self.df.isna().sum()
        return pd.DataFrame({
            "missing_count": missing,
            "missing_pct": missing / len(self.df)
        })

    def handle_missing_values(self) -> dict:
            """Fill numeric NaNs with median and categorical NaNs with mode.
            Returns a summary of imputed columns and number of values filled.
            """
            imputed_summary = {}
            for col in self.df.select_dtypes(include=np.number).columns:
                num_missing = self.df[col].isna().sum()
                self.df[col] = self.df[col].fillna(self.df[col].median())
                imputed_summary[col] = num_missing

            for col in self.df.select_dtypes(include="object").columns:
                num_missing = self.df[col].isna().sum()
                self.df[col] = self.df[col].fillna(self.df[col].mode()[0])
                imputed_summary[col] = num_missing

            return pd.DataFrame({"imputed_values_per_column": imputed_summary})

    def remove_duplicates(self) -> dict:
        """Drop duplicate rows. Returns number of duplicates removed."""
        num_duplicates = self.df.duplicated().sum()
        self.df = self.df.drop_duplicates()
        return {"duplicates_removed": num_duplicates}

    def fix_datetime_columns(self, cols: list[str]) -> pd.DataFrame:
        """
        Convert time columns to datetime and return their data types.
        """
        for col in cols:
            self.df[col] = pd.to_datetime(self.df[col], errors="coerce")
        
        return pd.DataFrame(self.df[cols].dtypes, columns=["dtype"])


class EDAProcessor:
    """Compute EDA metrics for fraud datasets (read-only)."""

    def __init__(self, df: pd.DataFrame):
        self.df = df

    def summary_statistics(self, numerical_cols: list[str]) -> pd.DataFrame:
        desc = self.df[numerical_cols].describe().T
        desc["skew"] = self.df[numerical_cols].skew()
        desc["kurtosis"] = self.df[numerical_cols].kurtosis()
        return desc

    def categorical_metrics(self, categorical_cols: list[str]) -> pd.DataFrame:
        summaries = {}
        for col in categorical_cols:
            counts = self.df[col].value_counts()
            probs = counts / counts.sum()
            summaries[col] = {
                "num_categories": counts.shape[0],
                "top_category_pct": probs.iloc[0],
                "entropy": entropy(probs)
            }
        return pd.DataFrame(summaries).T

    def class_distribution(self, target: str) -> pd.DataFrame:
        counts = self.df[target].value_counts().sort_index()
        return pd.DataFrame({
            "count": counts,
            "percentage": counts / counts.sum()
        })