import json

import numpy as np
import pandas as pd
import pytest

from synthschema import (
    GaussianCopulaSynthesizer,
    Inequality,
    OneOf,
    Range,
    certify,
    evaluate_quality,
    parse_rule,
)
from synthschema.cli import main
from synthschema.generators import generate_from_regex, infer_regex


def test_parse_rules():
    (c,) = parse_rule("ship_date >= order_date")
    assert isinstance(c, Inequality) and (c.low, c.high) == ("order_date", "ship_date")
    (c,) = parse_rule("a < b")
    assert (c.low, c.high, c.strict) == ("a", "b", True)
    (c,) = parse_rule("age BETWEEN 18 AND 90")
    assert isinstance(c, Range) and (c.min, c.max) == (18, 90)
    (c,) = parse_rule("status IN ('open', 'closed')")
    assert isinstance(c, OneOf) and c.values == ["open", "closed"]
    rules = parse_rule("x >= 0 AND x <= 10")
    assert [r.to_dict() for r in rules] == [
        {"type": "range", "column": "x", "min": 0},
        {"type": "range", "column": "x", "max": 10},
    ]
    with pytest.raises(ValueError):
        parse_rule("x LIKE 'a%'")


def test_regex_roundtrip():
    rng = np.random.default_rng(0)
    values = generate_from_regex("AB-[0-9]{4}-[A-Z]{2}", 500, rng)
    assert len(set(values)) == 500
    assert pd.Series(values).str.match(r"^AB-\d{4}-[A-Z]{2}$").all()
    assert infer_regex(values) == "AB\\-[0-9]{4}\\-[A-Z]{2}"
    widened = generate_from_regex("C[0-9]{2}", 1000, rng)
    assert len(set(widened)) == 1000


def test_quality_detects_bad_data(mixed_df):
    good = GaussianCopulaSynthesizer().fit(mixed_df).sample(800, seed=0)
    bad = good.copy()
    bad["x"] = np.random.default_rng(0).uniform(0, 1000, len(bad))
    assert evaluate_quality(mixed_df, good).score > evaluate_quality(mixed_df, bad).score + 0.05


def test_certificate_fails_on_copies(customers):
    df, md = customers
    cert = certify(df, df.copy(), md)
    assert not cert.passed
    assert any("privacy" in r for r in cert.reasons)
    fake = GaussianCopulaSynthesizer(md, "customers").fit(df).sample(1000, seed=0)
    ok = certify(df, fake, md)
    assert ok.passed, ok.reasons
    payload = json.loads(ok.to_json())
    assert payload["passed"] and len(payload["fingerprint"]) == 64


def test_cli(tmp_path, mixed_df, capsys):
    real = tmp_path / "real.csv"
    mixed_df.to_csv(real, index=False)
    out = tmp_path / "fake.csv"
    report = tmp_path / "cert.json"
    assert main(["synthesize", str(real), "-o", str(out), "-n", "300", "--seed", "1", "--report", str(report)]) == 0
    fake = pd.read_csv(out)
    assert len(fake) == 300 and list(fake.columns) == list(mixed_df.columns)
    assert json.loads(report.read_text())["quality"]["score"] > 0.8

    ddl = tmp_path / "s.sql"
    ddl.write_text("CREATE TABLE t (id INT PRIMARY KEY, v DECIMAL(5,2));")
    assert main(["from-ddl", str(ddl), "-o", str(tmp_path / "out"), "-n", "10"]) == 0
    assert len(pd.read_csv(tmp_path / "out" / "t.csv")) == 10

    assert main(["detect", str(real)]) == 0
    assert "columns" in capsys.readouterr().out
    assert main(["evaluate", str(real), str(out)]) in (0, 1)
