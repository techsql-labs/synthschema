"""synthschema quick start: single table, multi-table, schema-only and certification.

Run with:  python examples/quickstart.py
"""

from synthschema import (
    GaussianCopulaSynthesizer,
    MultiTableSynthesizer,
    SchemaSynthesizer,
    certify,
    evaluate_quality,
    load_demo,
)

# 1. Single table ---------------------------------------------------------------------
customers, customers_md = load_demo("customers")  # simulated demo data
synth = GaussianCopulaSynthesizer(customers_md)
synth.fit(customers)
fake_customers = synth.sample(1_000, seed=42)
print(fake_customers.head())
print(evaluate_quality(customers, fake_customers, customers_md))

# Conditional sampling: only corporate customers, correlations kept.
corporate = synth.sample(200, conditions={"segment": "Corporate"}, seed=1)
print("mean income, corporate:", round(corporate["income"].mean()))

# 2. Multi-table ----------------------------------------------------------------------
tables, metadata = load_demo("retail")
multi = MultiTableSynthesizer(metadata)
multi.fit(tables)
fake_tables = multi.sample(scale=1.0, seed=7)
print({name: len(df) for name, df in fake_tables.items()})

# 3. Certify before you share ---------------------------------------------------------
certificate = certify(tables, fake_tables, metadata)
print(certificate)

# 4. Schema only: no seed data at all -------------------------------------------------
ddl = """
CREATE TABLE stores (
    store_id INT PRIMARY KEY,
    region   VARCHAR(10) CHECK (region IN ('North', 'South', 'West')),
    opened   DATE
);
CREATE TABLE sales (
    sale_id  INT PRIMARY KEY,
    store_id INT REFERENCES stores(store_id),
    qty      INT CHECK (qty BETWEEN 1 AND 20),
    price    DECIMAL(8, 2) CHECK (price > 0)
);
"""
schema_data = SchemaSynthesizer.from_ddl(ddl, children_per_parent=25).sample(num_rows=10, seed=3)
print(schema_data["sales"].head())
