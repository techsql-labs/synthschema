import pandas as pd
import pytest

from synthschema import Column, Metadata
from synthschema.metadata import detect_column


def test_detects_sdtypes(mixed_df):
    md = Metadata.detect_from_dataframe(mixed_df, "mixed")
    t = md.get_table("mixed")
    assert t.primary_key == "id"
    assert t.columns["x"].sdtype == "numerical"
    assert t.columns["count"].representation == "int"
    assert t.columns["cat"].sdtype == "categorical"
    assert t.columns["flag"].sdtype == "boolean"
    assert t.columns["when"].sdtype == "datetime"


def test_detects_pii_dates_and_ids():
    df = pd.DataFrame({
        "email": [f"user{i}@example.com" for i in range(100)],
        "phone": [f"555-01{i:02d}" for i in range(100)],
        "first_name": ["Ann"] * 100,
        "sku": [f"SKU-{i:05d}" for i in range(100)],
        "day": [f"2024-01-{(i % 28) + 1:02d}" for i in range(100)],
    })
    md = Metadata.detect_from_dataframe(df)
    cols = md.get_table().columns
    assert cols["email"].sdtype == "email"
    assert cols["phone"].sdtype == "phone_number"
    assert cols["first_name"].sdtype == "first_name"
    assert cols["sku"].sdtype == "id"
    assert cols["sku"].regex_format == "SKU\\-[0-9]{5}"
    assert cols["day"].sdtype == "datetime"
    assert cols["day"].datetime_format == "%Y-%m-%d"


def test_detects_relationships(retail):
    data, _ = retail
    md = Metadata.detect_from_dataframes(data)
    assert md.tables["customers"].primary_key == "customer_id"
    assert md.tables["orders"].primary_key == "order_id"
    assert len(md.relationships) == 1
    rel = md.relationships[0]
    assert (rel.parent_table, rel.child_table, rel.child_foreign_key) == ("customers", "orders", "customer_id")
    assert md.topological_order() == ["customers", "orders"]


def test_roundtrip_json_and_yaml(tmp_path, retail):
    _, md = retail
    for name in ("bp.json", "bp.yaml"):
        path = tmp_path / name
        md.save(path)
        loaded = Metadata.load(path)
        assert loaded.to_dict() == md.to_dict()


def test_validation_errors():
    md = Metadata()
    md.add_table("a")
    md.add_column("a", "x", "numerical")
    md.tables["a"].primary_key = "missing"
    with pytest.raises(ValueError, match="primary key"):
        md.validate()


def test_unknown_sdtype():
    with pytest.raises(ValueError, match="Unknown sdtype"):
        Column("banana")


def test_numeric_fk_like_column_is_categorical_in_single_table():
    s = pd.Series([1, 2, 3] * 100, name="store_id")
    assert detect_column(s, "store_id").sdtype == "categorical"


def test_cycle_detection():
    md = Metadata()
    for t in ("a", "b"):
        md.add_table(t)
        md.add_column(t, "id", "id")
        md.add_column(t, "other_id", "id")
        md.set_primary_key(t, "id")
    md.add_relationship("a", "b", "other_id")
    md.add_relationship("b", "a", "other_id")
    with pytest.raises(ValueError, match="cycle"):
        md.topological_order()
