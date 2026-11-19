import numpy as np
import pandas as pd
import pytest

from synthschema import (
    FixedCombinations,
    GaussianCopulaSynthesizer,
    Inequality,
    Range,
    evaluate_privacy,
    evaluate_quality,
)


def test_fit_sample_keeps_shape_and_types(mixed_df):
    synth = GaussianCopulaSynthesizer().fit(mixed_df)
    fake = synth.sample(500, seed=0)
    assert list(fake.columns) == list(mixed_df.columns)
    assert len(fake) == 500
    assert fake["id"].is_unique
    assert fake["count"].dtype == mixed_df["count"].dtype
    assert fake["flag"].dtype == bool
    assert pd.api.types.is_datetime64_any_dtype(fake["when"])
    assert set(fake["cat"].unique()) <= {"high", "low"}


def test_statistical_balance(mixed_df):
    synth = GaussianCopulaSynthesizer().fit(mixed_df)
    fake = synth.sample(2000, seed=1)
    # marginals
    assert abs(fake["x"].mean() - mixed_df["x"].mean()) < 1.0
    assert abs(fake["x"].std() - mixed_df["x"].std()) < 1.0
    assert abs(fake["flag"].mean() - mixed_df["flag"].mean()) < 0.04
    assert abs(fake["y"].isna().mean() - mixed_df["y"].isna().mean()) < 0.03
    # correlations
    real_corr = mixed_df[["x", "y"]].corr().iloc[0, 1]
    fake_corr = fake[["x", "y"]].corr().iloc[0, 1]
    assert abs(real_corr - fake_corr) < 0.05
    # category depends on x
    assert fake.loc[fake["cat"] == "high", "x"].mean() > fake.loc[fake["cat"] == "low", "x"].mean() + 5
    report = evaluate_quality(mixed_df, fake)
    assert report.score > 0.9


def test_rounding_and_bounds(mixed_df):
    fake = GaussianCopulaSynthesizer().fit(mixed_df).sample(1000, seed=2)
    assert fake["x"].min() >= mixed_df["x"].min()
    assert fake["x"].max() <= mixed_df["x"].max()
    assert np.allclose(fake["x"], fake["x"].round(2))


def test_seed_reproducibility(mixed_df):
    synth = GaussianCopulaSynthesizer().fit(mixed_df)
    a, b = synth.sample(100, seed=7), synth.sample(100, seed=7)
    pd.testing.assert_frame_equal(a, b)
    c = synth.sample(100, seed=8)
    assert not a.equals(c)


@pytest.mark.parametrize("dist", ["norm", "beta", "truncnorm", "gamma", "uniform", "auto"])
def test_parametric_distributions(mixed_df, dist):
    fake = GaussianCopulaSynthesizer(default_distribution=dist).fit(mixed_df).sample(300, seed=0)
    assert fake["x"].notna().all()


def test_constraints_hold_on_every_row(retail):
    data, md = retail
    orders = data["orders"].drop(columns=["customer_id"])
    synth = GaussianCopulaSynthesizer(
        constraints=[
            "ship_date >= order_date",
            Inequality("discount", "amount"),
            Range("discount", min=0),
            FixedCombinations(["channel", "status"]),
        ]
    ).fit(orders)
    fake = synth.sample(2000, seed=3)
    assert (pd.to_datetime(fake["ship_date"]) >= pd.to_datetime(fake["order_date"])).all()
    assert (fake["discount"] <= fake["amount"]).all()
    assert (fake["discount"] >= 0).all()
    real_pairs = set(map(tuple, orders[["channel", "status"]].to_numpy()))
    assert set(map(tuple, fake[["channel", "status"]].to_numpy())) <= real_pairs
    assert fake["ship_date"].str.match(r"^\d{4}-\d{2}-\d{2}$").all()


def test_metadata_constraints_are_used(retail):
    data, md = retail
    from synthschema import Metadata

    single = Metadata.from_dict({"tables": {"orders": md.tables["orders"].to_dict()}})
    synth = GaussianCopulaSynthesizer(single, "orders").fit(data["orders"])
    fake = synth.sample(1000, seed=1)
    assert (pd.to_datetime(fake["ship_date"]) >= pd.to_datetime(fake["order_date"])).all()


def test_conditional_sampling_keeps_correlation(mixed_df):
    synth = GaussianCopulaSynthesizer().fit(mixed_df)
    hi = synth.sample(500, conditions={"cat": "high"}, seed=1)
    lo = synth.sample(500, conditions={"cat": "low"}, seed=1)
    assert (hi["cat"] == "high").all() and (lo["cat"] == "low").all()
    assert hi["x"].mean() > lo["x"].mean() + 5
    fixed = synth.sample(200, conditions={"x": 70.0}, seed=1)
    assert (fixed["x"] == 70.0).all()
    assert fixed["y"].mean() > mixed_df["y"].mean()


def test_conditions_on_constrained_column(retail):
    data, _ = retail
    orders = data["orders"].drop(columns=["customer_id"])
    synth = GaussianCopulaSynthesizer(constraints=[FixedCombinations(["channel", "status"])]).fit(orders)
    fake = synth.sample(50, conditions={"channel": "Phone"}, seed=0)
    assert (fake["channel"] == "Phone").all()


def test_sample_from_conditions(mixed_df):
    synth = GaussianCopulaSynthesizer().fit(mixed_df)
    cond = pd.DataFrame({"cat": ["high", "low", None], "x": [None, 40.0, None]})
    out = synth.sample_from_conditions(cond, seed=0)
    assert len(out) == 3
    assert out.loc[0, "cat"] == "high"
    assert out.loc[1, "x"] == 40.0


def test_pii_is_never_copied(customers):
    df, md = customers
    synth = GaussianCopulaSynthesizer(md, "customers").fit(df)
    fake = synth.sample(1000, seed=4)
    assert not set(fake["email"]) & set(df["email"])
    report = evaluate_privacy(df, fake, md)
    assert report.unique_match_rate == 0
    assert report.dcr_ratio is not None and report.dcr_ratio > 0.5


def test_save_and_load(tmp_path, mixed_df):
    synth = GaussianCopulaSynthesizer().fit(mixed_df)
    path = tmp_path / "model.pkl"
    synth.save(path)
    loaded = GaussianCopulaSynthesizer.load(path)
    pd.testing.assert_frame_equal(synth.sample(50, seed=1), loaded.sample(50, seed=1))


def test_errors(mixed_df):
    synth = GaussianCopulaSynthesizer()
    with pytest.raises(RuntimeError):
        synth.sample(10)
    with pytest.raises(ValueError):
        synth.fit(mixed_df.iloc[:0])
    synth.fit(mixed_df)
    with pytest.raises(KeyError):
        synth.sample(10, conditions={"nope": 1})


def test_contradictory_rules_raise(mixed_df):
    synth = GaussianCopulaSynthesizer(constraints=[Range("x", min=1000)]).fit(mixed_df)
    with pytest.raises(ValueError, match="Could not satisfy"):
        synth.sample(10, seed=0, max_tries=3)


def test_learned_correlations(mixed_df):
    corr = GaussianCopulaSynthesizer().fit(mixed_df).get_learned_correlations()
    assert corr.loc["x", "y"] > 0.8
