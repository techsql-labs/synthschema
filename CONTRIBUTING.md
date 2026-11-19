# Contributing to synthschema

Thanks for helping. Bug reports, ideas and pull requests are all welcome.

## Development setup

```bash
git clone https://github.com/techsql-labs/synthschema.git
cd synthschema
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest
ruff check src tests
```

## Ground rules

- **No real data.** Never commit real records, PII or extracts, including in tests or
  issues. Use `load_demo()` or generate fixtures in code.
- **Licences.** Dependencies must be OSI-approved and compatible with Apache-2.0.
  Source-available licences (for example the Business Source License) are not accepted.
- **Local by default.** The core library must not make network calls or send telemetry.
- **One interface.** New generators implement `fit(data)` and
  `sample(num_rows, conditions=None, seed=None)`. Their output must be reproducible from a seed.
- **Tests with every change.** Statistical behaviour needs a test that would fail if it broke.

## Pull requests

1. Open an issue first for anything larger than a small fix.
2. Keep pull requests focused, and add an entry under "Unreleased" in `CHANGELOG.md`.
3. Make sure `pytest` and `ruff check` pass.

By contributing you agree that your contributions are licensed under the Apache License 2.0.
