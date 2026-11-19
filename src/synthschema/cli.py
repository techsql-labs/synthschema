"""Command-line interface: ``synthschema --help``."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from ._version import __version__


def _read(path: str) -> pd.DataFrame:
    p = Path(path)
    if p.suffix.lower() == ".parquet":
        return pd.read_parquet(p)
    if p.suffix.lower() == ".json":
        return pd.read_json(p)
    return pd.read_csv(p)


def _write(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".parquet":
        df.to_parquet(path, index=False)
    elif path.suffix.lower() == ".json":
        df.to_json(path, orient="records", indent=2, date_format="iso")
    else:
        df.to_csv(path, index=False)


def _cmd_synthesize(args) -> int:
    from .evaluation import certify
    from .metadata import Metadata
    from .single_table import GaussianCopulaSynthesizer

    real = _read(args.input)
    md = Metadata.load(args.metadata) if args.metadata else None
    synth = GaussianCopulaSynthesizer(md, default_distribution=args.distribution)
    synth.fit(real)
    rows = args.rows if args.rows is not None else len(real)
    fake = synth.sample(rows, seed=args.seed)
    _write(fake, Path(args.output))
    print(f"Wrote {len(fake)} rows to {args.output}")
    if args.report:
        cert = certify(real, fake, synth.get_metadata())
        Path(args.report).write_text(cert.to_json())
        print(cert)
    return 0


def _cmd_from_ddl(args) -> int:
    from .schema import SchemaSynthesizer

    synth = SchemaSynthesizer.from_ddl(Path(args.ddl).read_text(), children_per_parent=args.children)
    tables = synth.sample(num_rows=args.rows, seed=args.seed)
    out = Path(args.output)
    for name, df in tables.items():
        _write(df, out / f"{name}.{args.format}")
        print(f"Wrote {len(df)} rows to {out / f'{name}.{args.format}'}")
    return 0


def _cmd_evaluate(args) -> int:
    from .evaluation import certify
    from .metadata import Metadata

    real, fake = _read(args.real), _read(args.synthetic)
    md = Metadata.load(args.metadata) if args.metadata else None
    cert = certify(real, fake, md, min_quality=args.min_quality)
    print(json.dumps(cert.to_dict(), indent=2, default=str))
    return 0 if cert.passed else 1


def _cmd_detect(args) -> int:
    from .metadata import Metadata

    md = Metadata.detect_from_dataframe(_read(args.input), args.table)
    if args.output:
        md.save(args.output)
        print(f"Wrote metadata to {args.output}")
    else:
        print(json.dumps(md.to_dict(), indent=2, default=str))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="synthschema", description="Schema-driven synthetic data generation.")
    parser.add_argument("--version", action="version", version=f"synthschema {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("synthesize", help="learn from a CSV/Parquet file and write synthetic rows")
    p.add_argument("input")
    p.add_argument("-o", "--output", required=True)
    p.add_argument("-n", "--rows", type=int)
    p.add_argument("-m", "--metadata", help="metadata JSON/YAML (detected if omitted)")
    p.add_argument("--seed", type=int)
    p.add_argument("--distribution", default="empirical")
    p.add_argument("--report", help="write a certificate JSON here")
    p.set_defaults(func=_cmd_synthesize)

    p = sub.add_parser("from-ddl", help="generate tables from SQL CREATE TABLE statements (no data)")
    p.add_argument("ddl")
    p.add_argument("-o", "--output", required=True, help="output directory")
    p.add_argument("-n", "--rows", type=int, default=100, help="rows per root table")
    p.add_argument("--children", type=float, default=3.0, help="child rows per parent row")
    p.add_argument("--format", choices=["csv", "parquet", "json"], default="csv")
    p.add_argument("--seed", type=int)
    p.set_defaults(func=_cmd_from_ddl)

    p = sub.add_parser("evaluate", help="certify synthetic data against real data")
    p.add_argument("real")
    p.add_argument("synthetic")
    p.add_argument("-m", "--metadata")
    p.add_argument("--min-quality", type=float, default=0.8)
    p.set_defaults(func=_cmd_evaluate)

    p = sub.add_parser("detect", help="infer metadata from a data file")
    p.add_argument("input")
    p.add_argument("-t", "--table", default="table")
    p.add_argument("-o", "--output", help="write JSON/YAML here")
    p.set_defaults(func=_cmd_detect)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
