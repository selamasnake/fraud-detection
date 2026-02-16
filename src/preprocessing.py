from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import logging

import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE


# Named Constants

DEFAULT_TEST_SIZE: float = 0.2
DEFAULT_RANDOM_STATE: int = 42


# Configuration Dataclass

@dataclass
class PreprocessingConfig:
    """
    Configuration for dataset splitting and balancing.
    
    Attributes:
        target_col: Name of target column.
        drop_cols: Columns to drop before modeling.
        test_size: Proportion of dataset used for testing.
        random_state: Seed for reproducibility.
        use_smote: Whether to apply SMOTE to training data.
    """
    target_col: str
    drop_cols: List[str] = field(default_factory=list)
    test_size: float = DEFAULT_TEST_SIZE
    random_state: int = DEFAULT_RANDOM_STATE
    use_smote: bool = True


# Preprocessor

class DataPreprocessor:
    """Handle train-test split and optional SMOTE balancing."""

    def __init__(self, df: pd.DataFrame, config: PreprocessingConfig):
        """
        Initialize DataPreprocessor.

        Args:
            df: Input dataset.
            config: Preprocessing configuration.
        """
        self.df = df.copy()
        self.config = config

    def _split_features_target(self) -> Tuple[pd.DataFrame, pd.Series]:
        """Separate features and target."""
        X = self.df.drop(columns=self.config.drop_cols + [self.config.target_col])
        y = self.df[self.config.target_col]
        return X, y

    def _log_distribution(self, y: pd.Series, label: str) -> None:
        """Log class distribution."""
        logging.info(f"{label} class distribution:\n{y.value_counts()}")

    def split(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Perform stratified train-test split.

        Returns:
            X_train, X_test, y_train, y_test
        """
        X, y = self._split_features_target()

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=self.config.test_size,
            random_state=self.config.random_state,
            stratify=y
        )

        self._log_distribution(y_train, "Training (before SMOTE)")
        self._log_distribution(y_test, "Test")

        return X_train, X_test, y_train, y_test

    def apply_smote(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Apply SMOTE to training data only.

        Returns:
            Balanced X_train and y_train.
        """
        if not self.config.use_smote:
            return X_train, y_train

        smote = SMOTE(random_state=self.config.random_state)
        X_balanced, y_balanced = smote.fit_resample(X_train, y_train)

        self._log_distribution(y_balanced, "Training (after SMOTE)")

        return X_balanced, y_balanced

    def prepare(
        self
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Full preprocessing pipeline:
        - Split
        - Apply SMOTE (train only)

        Returns:
            X_train, X_test, y_train, y_test
        """
        X_train, X_test, y_train, y_test = self.split()
        X_train, y_train = self.apply_smote(X_train, y_train)

        return X_train, X_test, y_train, y_test
