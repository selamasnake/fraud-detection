import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
import pytest
import pandas as pd
import numpy as np
from tempfile import NamedTemporaryFile
from data_processing import DataCleaner, EDAProcessor

# ----------------------
# Fixtures
# ----------------------
@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "Amount": [100, 200, 300, 400, 10000, np.nan],
        "Value": [10, 20, 30, 40, 500, np.nan],
        "Category": ["A", "A", "B", "B", "B", None],
        "Time": ["2023-01-01", "2023-01-02", "2023-01-03", None, "2023-01-05", "2023-01-06"]
    })

# ----------------------
# DataCleaner Tests
# ----------------------
def test_missing_values_summary(sample_df):
    cleaner = DataCleaner(sample_df)
    missing = cleaner.missing_values_summary()
    assert "missing_count" in missing.columns
    assert "missing_pct" in missing.columns
    assert missing.loc["Amount", "missing_count"] == 1
    assert missing.loc["Category", "missing_count"] == 1

def test_handle_missing_values(sample_df):
    cleaner = DataCleaner(sample_df)
    imputed_df = cleaner.handle_missing_values()
    # Check no NaNs remain
    assert cleaner.df.isna().sum().sum() == 0
    # Check imputed values count matches
    assert imputed_df.loc["Amount", "imputed_values_per_column"] == 1
    assert imputed_df.loc["Category", "imputed_values_per_column"] == 1

def test_remove_duplicates():
    df = pd.DataFrame({"A": [1, 1, 2], "B": [3, 3, 4]})
    cleaner = DataCleaner(df)
    result = cleaner.remove_duplicates()
    assert result["duplicates_removed"] == 1
    assert len(cleaner.df) == 2

def test_fix_datetime_columns(sample_df):
    cleaner = DataCleaner(sample_df)
    dt_df = cleaner.fix_datetime_columns(["Time"])
    assert dt_df["dtype"].iloc[0] == "datetime64[ns]"
    # NaN rows should remain NaT
    assert cleaner.df["Time"].isna().sum() == 1

# ----------------------
# EDAProcessor Tests
# ----------------------
def test_summary_statistics(sample_df):
    eda = EDAProcessor(sample_df)
    stats = eda.summary_statistics(["Amount", "Value"])
    assert "mean" in stats.columns
    assert stats.loc["Amount", "mean"] == pytest.approx(sample_df["Amount"].median(skipna=True) + 0) or stats.loc["Amount", "mean"] == pytest.approx(sample_df["Amount"].mean(skipna=True))
    assert "skew" in stats.columns
    assert "kurtosis" in stats.columns

def test_categorical_metrics(sample_df):
    eda = EDAProcessor(sample_df)
    metrics = eda.categorical_metrics(["Category"])
    assert metrics.loc["Category", "num_categories"] == 2  # After ignoring None
    assert 0 <= metrics.loc["Category", "entropy"] <= np.log(2) + 0.01

def test_class_distribution():
    df = pd.DataFrame({"Class": [0, 1, 1, 0, 1]})
    eda = EDAProcessor(df)
    dist = eda.class_distribution("Class")
    assert dist.loc[0, "count"] == 2
    assert dist.loc[1, "percentage"] == 3 / 5
