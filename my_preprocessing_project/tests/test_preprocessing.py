# tests/test_preprocessing.py

import pandas as pd
from pandas.testing import assert_series_equal

from src.preprocess import impute_numeric, drop_high_null_cols

def test_impute_numeric_median():
    df = pd.DataFrame({"A": [1, None, 3]})
    out = impute_numeric(df.copy(), method="median")
    # median of [1,3] is 2.0
    assert_series_equal(out["A"], pd.Series([1.0, 2.0, 3.0]))

def test_drop_high_null_cols():
    df = pd.DataFrame({
        "full": [1, 2, 3],
        "mostly_null": [None, None, 5]
    })
    # threshold 0.5 → drop any col with >50% nulls → 'mostly_null' dropped
    out = drop_high_null_cols(df, threshold=0.5)
    assert "mostly_null" not in out.columns
    assert "full" in out.columns
