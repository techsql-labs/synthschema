import pandas as pd

from synthschema import Metadata, SchemaSynthesizer, parse_ddl

DDL = """
-- retail schema
CREATE TABLE IF NOT EXISTS dbo.customers (
    customer_id   INT PRIMARY KEY,
    email         VARCHAR(120) NOT NULL,
    full_name     VARCHAR(80),
    segment       VARCHAR(20) CHECK (segment IN ('Consumer', 'Corporate', 'Small Business')),
    age           INT CHECK (age BETWEEN 18 AND 90),
    country       VARCHAR(40),
    created_at    TIMESTAMP
);

CREATE TABLE orders (
    order_id      BIGINT NOT NULL,
    customer_id   INT NOT NULL,
    order_date    DATE NOT NULL,
    ship_date     DATE,
    amount        DECIMAL(10, 2) CHECK (amount >= 0),
    status        ENUM('new', 'shipped', 'delivered'),
    is_gift       BOOLEAN,
    notes         TEXT,
    CONSTRAINT pk_orders PRIMARY KEY (order_id),
    CONSTRAINT fk_cust FOREIGN KEY (customer_id) REFERENCES customers (customer_id),
    CHECK (ship_date >= order_date)
);
"""


def test_parse_ddl():
    md = parse_ddl(DDL)
    assert set(md.tables) == {"customers", "orders"}
    c, o = md.tables["customers"], md.tables["orders"]
    assert c.primary_key == "customer_id" and o.primary_key == "order_id"
    assert c.columns["email"].sdtype == "email"
    assert c.columns["full_name"].sdtype == "name"
    assert c.columns["segment"].values == ["Consumer", "Corporate", "Small Business"]
    assert (c.columns["age"].min_value, c.columns["age"].max_value) == (18, 90)
    assert o.columns["amount"].decimals == 2
    assert o.columns["status"].values == ["new", "shipped", "delivered"]
    assert o.columns["order_date"].representation == "date"
    assert o.columns["notes"].sdtype == "text"
    assert md.relationships[0].child_foreign_key == "customer_id"


def test_schema_only_generation():
    data = SchemaSynthesizer.from_ddl(DDL).sample(num_rows=200, seed=1)
    c, o = data["customers"], data["orders"]
    assert len(c) == 200 and len(o) == 600
    assert c["customer_id"].is_unique and o["order_id"].is_unique
    assert o["customer_id"].isin(c["customer_id"]).all()
    assert c["age"].between(18, 90).all()
    assert set(c["segment"]) <= {"Consumer", "Corporate", "Small Business"}
    assert (o["amount"] >= 0).all()
    assert (pd.to_datetime(o["ship_date"]) >= pd.to_datetime(o["order_date"])).all()
    assert c["email"].str.contains("@").all()


def test_schema_only_is_reproducible():
    a = SchemaSynthesizer.from_ddl(DDL).sample(50, seed=9)
    b = SchemaSynthesizer.from_ddl(DDL).sample(50, seed=9)
    for name in a:
        pd.testing.assert_frame_equal(a[name], b[name])


def test_hand_built_blueprint():
    md = Metadata.from_dict({
        "tables": {
            "devices": {
                "primary_key": "device_id",
                "columns": {
                    "device_id": {"sdtype": "id", "regex_format": "DEV-[A-Z]{2}[0-9]{4}"},
                    "kind": {"sdtype": "categorical", "values": ["sensor", "gateway"], "weights": [0.9, 0.1]},
                    "temp": {"sdtype": "numerical", "min_value": -10, "max_value": 40, "decimals": 1},
                    "battery_min": {"sdtype": "numerical", "min_value": 0, "max_value": 50},
                    "battery_max": {"sdtype": "numerical", "min_value": 0, "max_value": 100},
                },
                "constraints": ["battery_max >= battery_min"],
            }
        }
    })
    df = SchemaSynthesizer(md).sample(1000, seed=0)["devices"]
    assert df["device_id"].str.match(r"^DEV-[A-Z]{2}\d{4}$").all()
    assert 0.85 < (df["kind"] == "sensor").mean() < 0.95
    assert df["temp"].between(-10, 40).all()
    assert (df["battery_max"] >= df["battery_min"]).all()
