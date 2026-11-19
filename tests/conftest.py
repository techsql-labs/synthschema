import numpy as np
import pandas as pd
import pytest

from synthschema import Metadata, load_demo


@pytest.fixture(scope="session")
def retail():
    return load_demo("retail")


@pytest.fixture(scope="session")
def customers():
    return load_demo("customers")


@pytest.fixture
def mixed_df():
    rng = np.random.default_rng(1)
    n = 800
    x = rng.normal(50, 10, n)
    y = 2 * x + rng.normal(0, 5, n)
    cat = np.where(x > 50, "high", "low")
    df = pd.DataFrame({
        "id": np.arange(n),
        "x": np.round(x, 2),
        "y": np.round(y, 2),
        "count": rng.poisson(3, n),
        "cat": cat,
        "flag": rng.random(n) < 0.3,
        "when": pd.Timestamp("2024-01-01") + pd.to_timedelta(rng.integers(0, 365, n), unit="D"),
    })
    df.loc[rng.random(n) < 0.1, "y"] = np.nan
    return df


@pytest.fixture
def mixed_metadata(mixed_df):
    return Metadata.detect_from_dataframe(mixed_df, "mixed")
