"""synthschema: open-source, schema-driven synthetic data generation.

Quick start::

    from synthschema import GaussianCopulaSynthesizer, evaluate_quality

    synth = GaussianCopulaSynthesizer()
    synth.fit(real_df)
    fake_df = synth.sample(1_000, seed=42)
    print(evaluate_quality(real_df, fake_df))
"""

from ._version import __version__
from .constraints import Custom, FixedCombinations, Inequality, OneOf, Range, parse_rule
from .datasets import load_demo
from .evaluation import (
    Certificate,
    DiagnosticReport,
    PrivacyReport,
    QualityReport,
    certify,
    evaluate_privacy,
    evaluate_quality,
    run_diagnostic,
)
from .metadata import Column, Metadata, Relationship, Table
from .multi_table import MultiTableSynthesizer
from .schema import SchemaSynthesizer, parse_ddl
from .single_table import GaussianCopulaSynthesizer

__all__ = [
    "__version__",
    "Metadata",
    "Table",
    "Column",
    "Relationship",
    "GaussianCopulaSynthesizer",
    "MultiTableSynthesizer",
    "SchemaSynthesizer",
    "parse_ddl",
    "Inequality",
    "Range",
    "OneOf",
    "FixedCombinations",
    "Custom",
    "parse_rule",
    "evaluate_quality",
    "run_diagnostic",
    "evaluate_privacy",
    "certify",
    "QualityReport",
    "DiagnosticReport",
    "PrivacyReport",
    "Certificate",
    "load_demo",
]
