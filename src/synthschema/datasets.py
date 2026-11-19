"""Demo datasets. These are simulated (no real people or companies) and deterministic."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .generators import _builtin_pii
from .metadata import Metadata


def _retail(n_customers: int = 1000, seed: int = 0) -> tuple[dict[str, pd.DataFrame], Metadata]:
    rng = np.random.default_rng(seed)
    n = n_customers
    region = rng.choice(["North", "South", "West"], size=n, p=[0.4, 0.35, 0.25])
    age = np.clip(np.round(rng.normal(42, 12, n)), 18, 85).astype(int)
    seg_p = np.where(age[:, None] > 45, [0.45, 0.35, 0.20], [0.65, 0.15, 0.20])
    segment = np.array(
        [rng.choice(["Consumer", "Corporate", "Small Business"], p=p) for p in seg_p], dtype=object
    )
    seg_mult = pd.Series(segment).map({"Consumer": 1.0, "Corporate": 1.8, "Small Business": 1.4}).to_numpy()
    income = np.round(np.exp(rng.normal(10.6, 0.35, n)) * seg_mult * (0.7 + age / 100), -2)
    signup = pd.Timestamp("2019-01-01") + pd.to_timedelta(rng.integers(0, 6 * 365, n), unit="D")
    first, last = _builtin_pii("first_name", n, rng), _builtin_pii("last_name", n, rng)
    email = [f"{fn.lower()}.{ln.lower()}{i}@example.com" for i, (fn, ln) in enumerate(zip(first, last, strict=True))]
    tenure_days = (pd.Timestamp("2025-01-01") - signup).days.to_numpy()
    active_p = 1 / (1 + np.exp(-(1.5 - tenure_days / 1000 + (income > 60000) * 0.5)))
    customers = pd.DataFrame({
        "customer_id": np.arange(1, n + 1),
        "first_name": first,
        "last_name": last,
        "email": email,
        "age": age,
        "region": region,
        "segment": segment,
        "income": income,
        "signup_date": signup.strftime("%Y-%m-%d"),
        "is_active": rng.random(n) < active_p,
    })

    lam = 1.5 + (income / income.mean()) * 1.5 + (segment == "Corporate") * 1.0
    counts = rng.poisson(lam)
    owner = np.repeat(np.arange(n), counts)
    m = len(owner)
    order_date = signup[owner] + pd.to_timedelta(rng.integers(0, 365, m), unit="D")
    ship_days = rng.choice([1, 2, 3, 4, 5, 7], size=m, p=[0.15, 0.3, 0.25, 0.15, 0.1, 0.05])
    channel = np.where(age[owner] < 40,
                       rng.choice(["Online", "Store", "Phone"], size=m, p=[0.7, 0.25, 0.05]),
                       rng.choice(["Online", "Store", "Phone"], size=m, p=[0.4, 0.45, 0.15]))
    amount = np.round(np.exp(rng.normal(3.6, 0.6, m)) * (income[owner] / 50000) ** 0.6 * seg_mult[owner], 2)
    discount = np.round(amount * rng.uniform(0, 0.3, m) * (rng.random(m) < 0.4), 2)
    status = rng.choice(["delivered", "shipped", "returned", "cancelled"], size=m, p=[0.8, 0.1, 0.06, 0.04])
    orders = pd.DataFrame({
        "order_id": [f"ORD-{i:06d}" for i in range(1, m + 1)],
        "customer_id": customers["customer_id"].to_numpy()[owner],
        "order_date": order_date.strftime("%Y-%m-%d"),
        "ship_date": (order_date + pd.to_timedelta(ship_days, unit="D")).strftime("%Y-%m-%d"),
        "channel": channel,
        "status": status,
        "amount": amount,
        "discount": discount,
    })

    md = Metadata()
    t = md.add_table("customers", primary_key="customer_id")
    md.add_column("customers", "customer_id", "id", representation="int")
    md.add_column("customers", "first_name", "first_name")
    md.add_column("customers", "last_name", "last_name")
    md.add_column("customers", "email", "email")
    md.add_column("customers", "age", "numerical", representation="int")
    md.add_column("customers", "region", "categorical")
    md.add_column("customers", "segment", "categorical")
    md.add_column("customers", "income", "numerical")
    md.add_column("customers", "signup_date", "datetime", datetime_format="%Y-%m-%d")
    md.add_column("customers", "is_active", "boolean")
    t.primary_key = "customer_id"
    md.add_table("orders", primary_key="order_id")
    md.add_column("orders", "order_id", "id", regex_format="ORD-[0-9]{6}")
    md.add_column("orders", "customer_id", "id", representation="int")
    md.add_column("orders", "order_date", "datetime", datetime_format="%Y-%m-%d")
    md.add_column("orders", "ship_date", "datetime", datetime_format="%Y-%m-%d")
    md.add_column("orders", "channel", "categorical")
    md.add_column("orders", "status", "categorical")
    md.add_column("orders", "amount", "numerical")
    md.add_column("orders", "discount", "numerical")
    md.add_relationship("customers", "orders", "customer_id")
    md.add_constraint("orders", "ship_date >= order_date")
    md.add_constraint("orders", "discount <= amount")
    md.add_constraint("orders", "discount >= 0")
    md.validate()
    return {"customers": customers, "orders": orders}, md


def load_demo(name: str = "retail", seed: int = 0):
    """Load a simulated demo dataset.

    * ``"retail"`` - ``(dict of DataFrames, Metadata)`` with customers and orders.
    * ``"customers"`` - ``(DataFrame, Metadata)`` with the customers table only.
    """
    data, md = _retail(seed=seed)
    if name == "retail":
        return data, md
    if name == "customers":
        single = Metadata.from_dict({"tables": {"customers": md.tables["customers"].to_dict()}})
        return data["customers"], single
    raise ValueError(f"Unknown demo {name!r}; choose 'retail' or 'customers'")
