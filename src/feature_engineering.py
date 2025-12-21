import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder

class FeatureEngineer:
    """Generate new features and transform datasets for fraud detection."""

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.scaler = None
        self.encoder = None

    # -----------------------
    # Time-Based Features
    # -----------------------
    def add_time_features_fraud(self, purchase_col="purchase_time", signup_col="signup_time"):
        """Add hour_of_day, day_of_week, and time_since_signup (in hours)."""
        self.df["hour_of_day"] = self.df[purchase_col].dt.hour
        self.df["day_of_week"] = self.df[purchase_col].dt.dayofweek
        self.df["time_since_signup"] = (
            self.df[purchase_col] - self.df[signup_col]
        ).dt.total_seconds() / 3600
        return self.df

    def add_time_features_cc(self, time_col="Time"):
        """Add hour_of_day and day_of_week from CreditCard Time column."""
        self.df["hour_of_day"] = (self.df[time_col] % 86400) // 3600
        self.df["day_of_week"] = (self.df[time_col] // 86400) % 7
        return self.df

    # -----------------------
    # Frequency / Velocity Features
    # -----------------------
    def add_transaction_counts(self, group_cols: list[str]):
        """Count number of transactions per group (e.g., user_id, device_id, country)."""
        for col in group_cols:
            count_col = f"{col}_tx_count"
            self.df[count_col] = self.df.groupby(col)[col].transform("count")
        return self.df

    # -----------------------
    # Binning / Transformation
    # -----------------------
    def bin_numerical(self, col: str, n_bins: int = 4, method="quantile"):
        """Create bins for a numerical column."""
        self.df[f"{col}_bin"] = pd.qcut(self.df[col], n_bins, labels=False)
        return self.df

    def scale_numerical(self, numerical_cols: list[str], method="standard"):
        """Scale numerical columns using StandardScaler or MinMaxScaler."""
        if method == "standard":
            self.scaler = StandardScaler()
        elif method == "minmax":
            self.scaler = MinMaxScaler()
        else:
            raise ValueError("method must be 'standard' or 'minmax'")

        self.df[numerical_cols] = self.scaler.fit_transform(self.df[numerical_cols])
        return self.df

    # -----------------------
    # Categorical Encoding
    # -----------------------
    def encode_categorical(self, categorical_cols: list[str], drop_first=True):
        """One-Hot encode categorical features."""
        self.encoder = OneHotEncoder(sparse_output=False, drop="first" if drop_first else None)
        encoded = self.encoder.fit_transform(self.df[categorical_cols])
        encoded_df = pd.DataFrame(
            encoded,
            columns=self.encoder.get_feature_names_out(categorical_cols),
            index=self.df.index
        )
        self.df = pd.concat([self.df.drop(columns=categorical_cols), encoded_df], axis=1)
        return self.df
