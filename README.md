# synthschema

**Open-source, schema-driven synthetic data for tabular and relational datasets.**

[![PyPI](https://img.shields.io/pypi/v/synthschema.svg)](https://pypi.org/project/synthschema/)
[![Python](https://img.shields.io/pypi/pyversions/synthschema.svg)](https://pypi.org/project/synthschema/)
[![CI](https://github.com/techsql-labs/synthschema/actions/workflows/ci.yml/badge.svg)](https://github.com/techsql-labs/synthschema/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)

synthschema learns the statistical shape of your data (each column's distribution and the
correlations between columns) and generates new rows that keep that balance, without copying
real records. It also works from a **schema alone**: give it SQL DDL and rules, and it
generates realistic data whose keys match across tables, with no seed data at all.

Every dataset can be **checked before you share it**: quality, structure, rules and privacy
roll up into a single pass/fail certificate.

```bash
pip install synthschema            # core: numpy, pandas, scipy
pip install "synthschema[all]"     # + Faker (realistic PII values) and PyYAML (YAML blueprints)
```

## Why synthschema

| | |
| --- | --- |
| **Statistical fidelity** | Exact empirical marginals plus a Gaussian copula keep distributions, category shares, null rates and correlations. |
| **Multi-table, keys intact** | Parents are generated first. Child counts per parent are modelled, and child rows are conditioned on their parent, so there are zero orphan keys and cross-table patterns survive. |
| **Schema-only mode** | `CREATE TABLE` statements → realistic, key-consistent data. `CHECK`, `ENUM`, `REFERENCES` and `NOT NULL` are respected. |
| **Rules on 100% of rows** | `ship_date >= order_date`, ranges, `IN` lists and fixed combinations. Rules are learned as transforms where possible, so the distribution is not distorted. |
| **Privacy by construction** | IDs, names, emails and other PII are generated, never learned. Real rows are never copied, and a privacy check proves it. |
| **Certify before release** | Quality, diagnostic and privacy reports roll up into a fingerprinted certificate with hard stops. |
| **Light and local** | Pure Python on numpy, pandas and scipy. No GPU and no network calls. Your data never leaves your machine. |

## Quick start

### One table

```python
from synthschema import GaussianCopulaSynthesizer, evaluate_quality, load_demo

real, metadata = load_demo("customers")     # simulated demo data; use any DataFrame

synth = GaussianCopulaSynthesizer(metadata)  # metadata is optional: it is detected if omitted
synth.fit(real)
fake = synth.sample(1_000, seed=42)          # same seed -> same rows

print(evaluate_quality(real, fake, metadata))
# QualityReport(score=95.5%; Column Shapes: 97.9%, Column Pair Trends: 93.2%)
```

Conditional sampling keeps correlations with the fixed values:

```python
corporate = synth.sample(200, conditions={"segment": "Corporate"})
```

### Several related tables

```python
from synthschema import Metadata, MultiTableSynthesizer, load_demo

tables, metadata = load_demo("retail")      # customers 1 --< orders
# or: metadata = Metadata.detect_from_dataframes(tables)

synth = MultiTableSynthesizer(metadata)
synth.fit(tables)
fake_tables = synth.sample(scale=1.0, seed=7)    # {"customers": df, "orders": df}
```

### Schema only: no data needed

```python
from synthschema import SchemaSynthesizer

ddl = """
CREATE TABLE customers (
    customer_id INT PRIMARY KEY,
    email       VARCHAR(120) NOT NULL,
    segment     VARCHAR(20) CHECK (segment IN ('Consumer', 'Corporate')),
    age         INT CHECK (age BETWEEN 18 AND 90)
);
CREATE TABLE orders (
    order_id    INT PRIMARY KEY,
    customer_id INT REFERENCES customers(customer_id),
    order_date  DATE,
    ship_date   DATE,
    amount      DECIMAL(10, 2) CHECK (amount >= 0),
    CHECK (ship_date >= order_date)
);
"""
data = SchemaSynthesizer.from_ddl(ddl).sample(num_rows=500, seed=1)
```

### Rules

```python
from synthschema import FixedCombinations, GaussianCopulaSynthesizer, Inequality, Range

synth = GaussianCopulaSynthesizer(constraints=[
    "ship_date >= order_date",                 # SQL-like strings...
    "discount BETWEEN 0 AND 0.4",
    Inequality(low="discount", high="amount"), # ...or objects
    FixedCombinations(["country", "currency"]),
])
```

Rules can also live in the metadata blueprint, so they are declared once and used by every
synthesizer and by the diagnostic check.

### Certify before you share

```python
from synthschema import certify

certificate = certify(tables, fake_tables, metadata)
print(certificate)
# Certificate(PASSED; quality=97.1%, diagnostic=ok, reasons=[])
certificate.save("certificate.json")
```

Hard stops: any broken primary key, foreign key or rule; any synthetic row that copies a
real row that is unique in the real data; or quality below the threshold (default 80%).

### Command line

```bash
synthschema synthesize customers.csv -o synthetic.csv -n 10000 --seed 1 --report cert.json
synthschema from-ddl schema.sql -o out/ -n 1000
synthschema evaluate real.csv synthetic.csv          # exit code 1 if the certificate fails
synthschema detect customers.csv -o blueprint.yaml
```

## Metadata blueprints

Metadata is the single description of your dataset. Detect it, edit it, and keep it in Git:

```yaml
tables:
  customers:
    primary_key: customer_id
    columns:
      customer_id: {sdtype: id, representation: int}
      email: {sdtype: email}
      age: {sdtype: numerical, representation: int}
      segment: {sdtype: categorical}
      signup_date: {sdtype: datetime, datetime_format: "%Y-%m-%d"}
  orders:
    primary_key: order_id
    columns:
      order_id: {sdtype: id, regex_format: "ORD-[0-9]{6}"}
      customer_id: {sdtype: id}
      order_date: {sdtype: datetime, datetime_format: "%Y-%m-%d"}
      ship_date: {sdtype: datetime, datetime_format: "%Y-%m-%d"}
      amount: {sdtype: numerical}
    constraints:
      - "ship_date >= order_date"
relationships:
  - {parent_table: customers, parent_primary_key: customer_id, child_table: orders, child_foreign_key: customer_id}
```

**Column types (`sdtype`):**

- **Modelled:** `numerical`, `categorical`, `boolean`, `datetime`.
- **Generated, never learned:** `id`, `text`.
- **Generated PII:** `email`, `phone_number`, `name`, `first_name`, `last_name`, `address`,
  `street_address`, `ssn`, `credit_card_number`, `ip_address`, `url`, `company`, `postcode`.

## How it works

1. **Profile.** Each column gets a type and a reversible transform: empirical marginal,
   null indicator, rounding and min/max for numbers; frequency intervals for categories;
   epoch seconds plus original format for dates.
2. **Learn.** Every value is mapped to a normal score. A Gaussian copula learns the
   correlation matrix between all columns.
3. **Constrain.** Rules become transforms during training where possible (an inequality
   is learned as a non-negative gap). Every row is checked after sampling, and any failing
   row is redrawn.
4. **Generate.** Rows are drawn from the copula (or from its exact conditional
   distribution when values are fixed), mapped back to data values, and given freshly
   generated keys and PII.
5. **Relate.** For multiple tables, each parent also learns its child count. Children are
   sampled conditioned on their parent's values, in topological order.
6. **Certify.** Quality (KS / TVD shapes, correlation and contingency similarity,
   cardinality), diagnostics (keys, rules, ranges) and privacy (exact copies, distance to
   closest record) become one release decision.

## Roadmap

synthschema 0.1 is the community foundation. Planned next:

- More generators behind the same `fit` / `sample` interface: CTGAN/TVAE-style deep models,
  tabular diffusion and sequential (time-series) models.
- Relational latent diffusion for deep schemas with 50+ tables.
- Causal what-if scenarios (`do(price = price * 1.1)` propagating across tables).
- Consumer fidelity contracts: register KPIs and SQL queries and certify their parity.
- Differential-privacy mode and a cumulative privacy budget across releases.
- Context-aware LLM text columns and Q&A / evaluation sets, through a model endpoint you
  configure.
- Spark and Databricks scale-out as optional extras.

Ideas and pull requests are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

Apache License 2.0. See [LICENSE](LICENSE).

synthschema is built and maintained by **TechSQL Consulting Services**. For enterprise
support, custom generators or integration work, open an issue on GitHub and we will be in touch.
