# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/).

## [0.1.0] - 2026-11-18

First community release.

### Added
- `Metadata` blueprint: tables, columns, primary keys, relationships and rules. It can be
  detected from DataFrames, parsed from SQL DDL, or saved and loaded as JSON or YAML.
- `GaussianCopulaSynthesizer` for single tables:
  - Empirical or parametric marginals (`norm`, `beta`, `truncnorm`, `gamma`, `uniform`, `auto`).
  - Learned null rates, rounding and min/max.
  - Exact conditional sampling and seeded reproducibility.
- `MultiTableSynthesizer`:
  - Parent-first generation with learned child counts and parent-conditioned children.
  - Weighted secondary foreign keys and self-references.
- `SchemaSynthesizer`: data from DDL or metadata hints alone (no seed data).
- Constraints: `Inequality`, `Range`, `OneOf`, `FixedCombinations`, `Custom`, and SQL-like
  rule strings.
- Generators for IDs (learned regex patterns), PII (Faker when installed, built-in otherwise)
  and free text. Real values of these columns are never learned or copied.
- Evaluation: `evaluate_quality`, `run_diagnostic`, `evaluate_privacy` and `certify`, which
  produces a fingerprinted pass/fail certificate with hard stops.
- `synthschema` command line: `synthesize`, `from-ddl`, `evaluate`, `detect`.
- Simulated demo datasets via `load_demo`.

[0.1.0]: https://github.com/techsql-labs/synthschema/releases/tag/v0.1.0
