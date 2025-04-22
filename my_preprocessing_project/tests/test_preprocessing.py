# tests/test_preprocessing.py

import pandas as pd
import numpy as np
import pytest
from pandas.testing import assert_series_equal

from src.preprocess import (
    drop_high_null_cols,
    impute_numeric,
    impute_categorical,
    handle_missing_values,
    clean_column_names,
    apply_recategorization,
)

@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "A": [1, np.nan, 3, np.nan],
        "B": ["x", None, "y", None],
        "C": [None, None, None, None],
        "D E": [10, 20, 30, 40],
    })

def test_drop_high_null_cols(sample_df):
    df = sample_df.copy()
    # threshold 0.5 → drop any col with >50% nulls → 'C' and 'B' dropped
    out = drop_high_null_cols(df, drop_threshold=0.5)
    assert "C" not in out.columns
    assert "B" in out.columns
    assert "A" in out.columns

def test_impute_numeric_median():
    df = pd.DataFrame({"A": [1, None, 3]})
    out = impute_numeric(df.copy(), method="median")
    # median of [1,3] is 2.0; name the expected Series "A"
    expected = pd.Series([1.0, 2.0, 3.0], name="A")
    assert_series_equal(out["A"], expected)

def test_impute_categorical_mode():
    df = pd.DataFrame({"B": ["x", None, "y", None]})
    out = impute_categorical(df.copy(), method="mode")
    # mode of ["x","y"] is "x"
    expected = pd.Series(["x", "x", "y", "x"], name="B")
    assert_series_equal(out["B"], expected)

def test_handle_missing_values_integration(monkeypatch, sample_df):
    # Monkey‑patch get_preprocessing_config to return controlled thresholds
    from src.utils.config import get_preprocessing_config
    monkeypatch.setattr(
        'src.utils.config.get_preprocessing_config',
        lambda: {"missing_values": {"drop_threshold": 0.5, "imputation_method": "median"}}
    )
    df = handle_missing_values(sample_df.copy())
    # With drop_threshold=0.5, columns B and C should be dropped
    assert "B" in df.columns and "C" not in df.columns
    # Column A should be fully imputed (no NaNs)
    assert df["A"].isnull().sum() == 0

def test_clean_column_names():
    df = pd.DataFrame({ "X Y": [1], " Z ": [2] })
    out = clean_column_names(df)
    assert "x_y" in out.columns and "z" in out.columns

def test_apply_recategorization(monkeypatch):
    df = pd.DataFrame({"education": ["PhD","Basic","Unknown"]})
    # Monkey‑patch config for recategorization
    from src.utils.config import get_preprocessing_config
    monkeypatch.setattr(
        'src.utils.config.get_preprocessing_config',
        lambda: {"feature_engineering": {"recategorization": {"education": {"PhD":"Post Graduate","Basic":"Pre Graduate"}}}}
    )
    out = apply_recategorization(df.copy())
    assert out.loc[0,"education"] == "Post Graduate"
    assert out.loc[1,"education"] == "Pre Graduate"
    # “Unknown” stays unchanged
    assert out.loc[2,"education"] == "Unknown"

