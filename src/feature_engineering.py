import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
import logging

# Optional: minimal logging configuration for module
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# Named constants
SECONDS_IN_HOUR = 3600

@dataclass
class FeatureConfig:
    """
    Configuration for feature engineering.

    Attributes:
        time_cols: Columns with datetime info (purchase, signup, etc.).
        group_cols: Columns to count transactions per group (user_id, device_id, country, etc.).
        numerical_bins: Dict of {col: n_bins} for binning numeric columns.
        scaling_method: 'standard' or 'minmax' for numerical scaling.
        categorical_cols: Columns to encode.
        drop_first: Drop first category for one-hot encoding.
        purchase_col: Name of purchase datetime column.
        signup_col: Name of signup datetime column.
    """
    time_cols: List[str] = field(default_factory=list)
    group_cols: List[str] = field(default_factory=list)
    numerical_bins: dict = field(default_factory=dict)
    scaling_method: str = "standard"
    categorical_cols: List[str] = field(default_factory=list)
    drop_first: bool = True
    purchase_col: str = "purchase_time"
    signup_col: str = "signup_time"

class FeatureEngineer:
    """Generate new features and transform datasets for fraud detection."""

    def __init__(self, df: pd.DataFrame, config: Optional[FeatureConfig] = None):
        """
        Initialize FeatureEngineer.

        Args:
            df: Transaction DataFrame.
            config: Optional FeatureConfig object. Defaults used if None.
        """
        self.df = df.copy()
        self.config = config or FeatureConfig()
        self.scaler: Optional[StandardScaler | MinMaxScaler] = None
        self.encoder: Optional[OneHotEncoder] = None

    # -----------------------
    # Time-Based Features
    # -----------------------
    def add_time_features_fraud(self, purchase_col: Optional[str] = None, signup_col: Optional[str] = None) -> pd.DataFrame:
        """
        Add hour_of_day, day_of_week, and time_since_signup (in hours).

        Args:
            purchase_col: Purchase datetime column. Defaults to config.purchase_col.
            signup_col: Signup datetime column. Defaults to config.signup_col.

        Returns:
            pd.DataFrame with new time features added.
        """
        purchase_col = purchase_col or self.config.purchase_col
        signup_col = signup_col or self.config.signup_col

        self.df["hour_of_day"] = self.df[purchase_col].dt.hour
        self.df["day_of_week"] = self.df[purchase_col].dt.dayofweek
        self.df["time_since_signup"] = (
            self.df[purchase_col] - self.df[signup_col]
        ).dt.total_seconds() / SECONDS_IN_HOUR

        logging.info(f"Added time features: hour_of_day, day_of_week, time_since_signup")
        return self.df

    def add_time_features_cc(self, time_col: str = "Time") -> pd.DataFrame:
        """
        Add hour_of_day and day_of_week from CreditCard Time column.

        Args:
            time_col: Column representing seconds since epoch or day.

        Returns:
            pd.DataFrame with new time features added.
        """
        self.df["hour_of_day"] = (self.df[time_col] % 86400) // 3600
        self.df["day_of_week"] = (self.df[time_col] // 86400) % 7

        logging.info(f"Added CC time features: hour_of_day, day_of_week from {time_col}")
        return self.df

    # -----------------------
    # Frequency / Velocity Features
    # -----------------------
    def add_transaction_counts(self, group_cols: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Count number of transactions per group (e.g., user_id, device_id, country).

        Args:
            group_cols: Columns to group by. Defaults to config.group_cols.

        Returns:
           pd.DataFrame with transaction count features added.
        """
        group_cols = group_cols or self.config.group_cols

        for col in group_cols:
            count_col = f"{col}_tx_count"
            self.df[count_col] = self.df.groupby(col)[col].transform("count")

        logging.info(f"Added transaction count features for columns: {group_cols}")
        return self.df

    # -----------------------
    # Binning / Transformation
    # -----------------------
    def bin_numerical(self) -> pd.DataFrame:
        """
        Bin numeric columns according to configuration (config.numerical_bins).

        Returns:
            pd.DataFrame with binned columns added.
        """
        for col, n_bins in self.config.numerical_bins.items():
            self.df[f"{col}_bin"] = pd.qcut(self.df[col], n_bins, labels=False, duplicates='drop')

        logging.info(f"Binned numerical columns: {list(self.config.numerical_bins.keys())}")
        return self.df

    def scale_numerical(self, numerical_cols: List[str], method: str = "standard") -> pd.DataFrame:
        """
        Scale numerical columns using StandardScaler or MinMaxScaler.

        Args:
            numerical_cols: List of numeric columns to scale.
            method: 'standard' or 'minmax'.

        Returns:
            pd.DataFrame with scaled columns.
        """
        if method == "standard":
            self.scaler = StandardScaler()
        elif method == "minmax":
            self.scaler = MinMaxScaler()
        else:
            raise ValueError("method must be 'standard' or 'minmax'")

        # Sanity check
        assert not self.df[numerical_cols].isna().any().any(), "NaNs detected before scaling"

        self.df[numerical_cols] = self.scaler.fit_transform(self.df[numerical_cols])

        logging.info(f"Scaled columns {numerical_cols} using {method} scaler")
        return self.df

    # -----------------------
    # Categorical Encoding
    # -----------------------
    def encode_categorical(self, categorical_cols: List[str], drop_first: bool = True) -> pd.DataFrame:
        """
        One-Hot encode categorical features.

        Args:
            categorical_cols: List of categorical columns to encode.
            drop_first: Drop first category to avoid multicollinearity.

        Returns:
            pd.DataFrame with categorical features encoded.
        """
        self.encoder = OneHotEncoder(sparse_output=False, drop="first" if drop_first else None)
        encoded = self.encoder.fit_transform(self.df[categorical_cols])
        encoded_df = pd.DataFrame(
            encoded,
            columns=self.encoder.get_feature_names_out(categorical_cols),
            index=self.df.index
        )
        self.df = pd.concat([self.df.drop(columns=categorical_cols), encoded_df], axis=1)

        logging.info(f"Encoded categorical columns: {categorical_cols}")
        return self.df
