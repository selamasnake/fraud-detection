import os
import pandas as pd
import numpy as np
from scipy.stats import entropy
from dataclasses import dataclass, field
from typing import List
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)

@dataclass
class CleaningConfig:
    """Stores settings for DataCleaner operations."""
    datetime_cols: List[str] = field(default_factory=list)  # columns to convert
    numeric_fill_strategy: str = "median"  # how to fill numeric NaNs
    categorical_fill_strategy: str = "mode"  # how to fill categorical NaNs
    min_datetime_conversion_pct: float = 0.9  # threshold for datetime conversion

class DataLoader:
    """Load and save CSV datasets.

    Design decision:
        - Use low_memory=False to prevent dtype inference issues for large datasets.
    """

    def __init__(self, path: str | None = None):
        """
        Initialize DataLoader with optional file path.

        Args:
            path (str | None): Path to the CSV file. Can be set later in load_data.
        """
        self.path = path
        self.df: pd.DataFrame | None = None


    def load_data(self, path: str | None = None) -> pd.DataFrame:
        """
        Load CSV into a pandas DataFrame with file existence checks.

        Args:
            path (str | None): Optional path to CSV file. Overrides self.path if provided.

        Returns:
            pd.DataFrame: Loaded DataFrame.
        """
        file_path = path or self.path
        if not file_path:
            raise ValueError("No file path specified.")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        logging.info(f"Loading data from {file_path}")
        self.df = pd.read_csv(file_path, low_memory=False)
        logging.info(f"Loaded {len(self.df)} rows and {len(self.df.columns)} columns")
        return self.df

    def save_data(self, output_path: str) -> None:
        """
        Save the DataFrame to a CSV file.

        Args:
            output_path (str): File path to save the CSV.

        Returns:
            None
        """
        
        if self.df is None:
            raise ValueError("No data loaded.")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        self.df.to_csv(output_path, index=False)
        logging.info(f"Saved DataFrame to {output_path} ({len(self.df)} rows)")


class DataCleaner:
    """Clean data: handle missing values, duplicates, and datetime columns.

    Design decisions:
        - Numeric NaNs filled with median to preserve central tendency.
        - Categorical NaNs filled with mode to retain most common category.
        - Fix datetime columns with errors='coerce' to avoid breaking pipeline.
    """

    def __init__(self, df: pd.DataFrame, config: CleaningConfig | None = None):
        """
        Initialize DataCleaner with a copy of the DataFrame and configuration.

        Args:
            df (pd.DataFrame): DataFrame to clean.
            config (CleaningConfig | None): Optional configuration for cleaning.
        """
        self.df = df.copy()
        self.config = config or CleaningConfig()

    def data_overview(self) -> dict:
        """
        Print info about dataset: number of rows, columns, and dtypes.

        Returns:
            dict: Dataset info (via pandas .info().
        """
        return self.df.info()

    def missing_values_summary(self) -> pd.DataFrame:
        """
        Summarize missing values for all columns.

        Returns:
            pd.DataFrame: Columns with count and percentage of missing values.
        """
        missing = self.df.isna().sum()
        summary = pd.DataFrame({
            "missing_count": missing,
            "missing_pct": missing / len(self.df)
        })
        return summary

    def handle_missing_values(self) -> pd.DataFrame:
        """
        Impute missing values: median for numeric, mode for categorical.

        Returns:
            pd.DataFrame: Table summarizing number of values imputed per column.
        """
        imputed_summary = {}

        # Numeric columns
        for col in self.df.select_dtypes(include=np.number).columns:
            num_missing = self.df[col].isna().sum()
            if num_missing > 0:
                self.df[col] = self.df[col].fillna(self.df[col].median())
            imputed_summary[col] = num_missing

        # Categorical columns
        for col in self.df.select_dtypes(include="object").columns:
            num_missing = self.df[col].isna().sum()
            if num_missing > 0:
                self.df[col] = self.df[col].fillna(self.df[col].mode()[0])
            imputed_summary[col] = num_missing

        return pd.DataFrame({"imputed_values_per_column": imputed_summary})


    def remove_duplicates(self) -> dict:
        """
        Remove duplicate rows in the DataFrame.

        Returns:
            dict: Number of duplicate rows removed.
        """
        num_duplicates = self.df.duplicated().sum()
        if num_duplicates > 0:
            logging.info(f"Removing {num_duplicates} duplicate rows")
            self.df.drop_duplicates(inplace=True)
        return {"duplicates_removed": num_duplicates}
    

    def fix_datetime_columns(self, cols: list[str] | None = None) -> pd.DataFrame:
        """
        Convert specified columns to datetime. Uses config defaults if no columns passed.

        Args:
            cols (list[str] | None): Columns to convert. Defaults to config.datetime_cols.

        Returns:
            pd.DataFrame: dtypes of converted columns.
        """
        cols = cols or self.config.datetime_cols
        for col in cols:
            self.df[col] = pd.to_datetime(self.df[col], errors="coerce")
            num_nulls = self.df[col].isna().sum()
            total_rows = len(self.df)
            success_pct = (total_rows - num_nulls) / total_rows
            if success_pct < self.config.min_datetime_conversion_pct:
                logging.warning(
                    f"{num_nulls} rows could not be converted in '{col}' "
                    f"({success_pct*100:.2f}% success)"
                )
        return pd.DataFrame(self.df[cols].dtypes, columns=["dtype"])


class EDAProcessor:
    """Compute EDA metrics for fraud datasets.

    Design decisions:
        - Added entropy to quantify diversity of categorical features.
        - Skew/kurtosis added to detect non-normal distributions.
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df

    def summary_statistics(self, numerical_cols: list[str]) -> pd.DataFrame:
        """
        Compute descriptive statistics for numerical columns.

        Args:
            numerical_cols (list[str]): List of numerical column names.

        Returns:
            pd.DataFrame: Table with count, mean, std, min, max, skew, and kurtosis.
        """
        desc = self.df[numerical_cols].describe().T
        desc["skew"] = self.df[numerical_cols].skew()
        desc["kurtosis"] = self.df[numerical_cols].kurtosis()
        return desc

    def categorical_metrics(self, categorical_cols: list[str]) -> pd.DataFrame:
        """
        Compute categorical metrics including number of categories, 
        top category percentage, and entropy.

        Args:
            categorical_cols (list[str]): List of categorical column names.

        Returns:
            pd.DataFrame: Table with metrics for each categorical column.
        """
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
        """
        Compute counts and percentages for each class in the target column.

        Args:
            target (str): Target column name.

        Returns:
            pd.DataFrame: Table with counts and percentages per class.
        """
        counts = self.df[target].value_counts().sort_index()
        return pd.DataFrame({
            "count": counts,
            "percentage": counts / counts.sum()
        })
    
    def compute_country_fraud_stats(
        self, country_col: str = "country", target_col: str = "class", min_count: int = 1000
    ) -> pd.DataFrame:
        """
        Calculate transaction count and average fraud rate per country, filtered by min transactions.

        Args:
            country_col (str): Column containing country names.
            target_col (str): Target variable column.
            min_count (int): Minimum transactions for reliable fraud rate.

        Returns:
            pd.DataFrame: DataFrame with 'count' and 'mean' columns.
        """
        stats = self.df.groupby(country_col)[target_col].agg(['count', 'mean'])
        reliable_stats = stats[stats['count'] > min_count].sort_values('mean', ascending=False)
        return reliable_stats