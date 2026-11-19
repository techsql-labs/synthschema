import numpy as np
import pandas as pd

from synthschema import Metadata, MultiTableSynthesizer, certify, evaluate_quality, run_diagnostic


def test_multi_table_integrity_and_fidelity(retail):
    data, md = retail
    synth = MultiTableSynthesizer(md).fit(data)
    out = synth.sample(seed=0)
    assert set(out) == {"customers", "orders"}
    c, o = out["customers"], out["orders"]
    assert len(c) == len(data["customers"])
    assert 0.8 * len(data["orders"]) < len(o) < 1.2 * len(data["orders"])
    assert c["customer_id"].is_unique and o["order_id"].is_unique
    assert o["customer_id"].isin(c["customer_id"]).all()
    assert (pd.to_datetime(o["ship_date"]) >= pd.to_datetime(o["order_date"])).all()
    assert run_diagnostic(data, out, md).passed
    assert evaluate_quality(data, out, md).score > 0.9


def test_cross_table_correlation_is_kept(retail):
    data, md = retail
    out = MultiTableSynthesizer(md).fit(data).sample(seed=1)

    def corr(tables):
        spend = tables["orders"].groupby("customer_id")["amount"].mean()
        income = tables["customers"].set_index("customer_id")["income"]
        joined = pd.concat([spend, income], axis=1, join="inner")
        return joined.corr().iloc[0, 1]

    assert corr(out) > 0.5 * corr(data) > 0


def test_child_counts_follow_parent(retail):
    data, md = retail
    out = MultiTableSynthesizer(md).fit(data).sample(seed=2)

    def mean_orders(tables, segment):
        counts = tables["orders"]["customer_id"].value_counts()
        cust = tables["customers"]
        ids = cust.loc[cust["segment"] == segment, "customer_id"]
        return ids.map(counts).fillna(0).mean()

    assert mean_orders(out, "Corporate") > mean_orders(out, "Consumer")


def test_scale_and_num_rows(retail):
    data, md = retail
    synth = MultiTableSynthesizer(md).fit(data)
    half = synth.sample(scale=0.5, seed=0)
    assert len(half["customers"]) == 500
    exact = synth.sample(num_rows={"customers": 42}, seed=0)
    assert len(exact["customers"]) == 42
    assert exact["orders"]["customer_id"].isin(exact["customers"]["customer_id"]).all()


def test_three_levels_secondary_parent_and_self_reference():
    rng = np.random.default_rng(0)
    stores = pd.DataFrame({"store_id": range(1, 21), "region": rng.choice(["N", "S"], 20)})
    products = pd.DataFrame({"product_id": range(1, 51), "price": rng.uniform(1, 100, 50).round(2)})
    employees = pd.DataFrame({
        "employee_id": range(1, 41),
        "store_id": rng.integers(1, 21, 40),
        "manager_id": [None] + list(rng.integers(1, 40, 39)),
        "salary": rng.normal(50000, 8000, 40).round(),
    })
    sales = pd.DataFrame({
        "sale_id": range(1, 401),
        "employee_id": rng.integers(1, 41, 400),
        "product_id": rng.integers(1, 51, 400),
        "qty": rng.integers(1, 10, 400),
    })
    data = {"stores": stores, "products": products, "employees": employees, "sales": sales}
    md = Metadata.detect_from_dataframes(data)
    md.update_column("employees", "manager_id", sdtype="id")
    md.add_relationship("employees", "employees", "manager_id")
    out = MultiTableSynthesizer(md).fit(data).sample(seed=5)
    assert out["employees"]["store_id"].isin(out["stores"]["store_id"]).all()
    assert out["sales"]["employee_id"].isin(out["employees"]["employee_id"]).all()
    assert out["sales"]["product_id"].isin(out["products"]["product_id"]).all()
    managers = out["employees"]["manager_id"].dropna()
    assert managers.isin(out["employees"]["employee_id"]).all()
    assert certify(data, out, md, min_quality=0.5).diagnostic.passed


def test_save_load(tmp_path, retail):
    data, md = retail
    synth = MultiTableSynthesizer(md).fit(data)
    synth.save(tmp_path / "m.pkl")
    loaded = MultiTableSynthesizer.load(tmp_path / "m.pkl")
    a, b = synth.sample(seed=3), loaded.sample(seed=3)
    for name in a:
        pd.testing.assert_frame_equal(a[name], b[name])
