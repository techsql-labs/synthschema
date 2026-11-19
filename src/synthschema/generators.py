"""Value generators for columns that are never learned: keys, PII and free text.

Real values of these columns never reach a model and are never copied to the output.
"""

from __future__ import annotations

import re
from collections.abc import Iterable

import numpy as np
import pandas as pd

from .metadata import Column

# ---------------------------------------------------------------------- regex patterns

_CLASS_CHARS = {
    "0-9": "0123456789",
    "A-Z": "ABCDEFGHIJKLMNOPQRSTUVWXYZ",
    "a-z": "abcdefghijklmnopqrstuvwxyz",
}
_TOKEN_RE = re.compile(r"\[([^\]]+)\]|\\d|\\w|\\(.)|(.)")
_QUANT_RE = re.compile(r"\{(\d+)(?:,(\d+))?\}")


def _expand_class(spec: str) -> str:
    chars: list[str] = []
    i = 0
    while i < len(spec):
        if i + 2 < len(spec) and spec[i + 1] == "-":
            start, end = ord(spec[i]), ord(spec[i + 2])
            chars.extend(chr(c) for c in range(start, end + 1))
            i += 3
        else:
            chars.append(spec[i])
            i += 1
    return "".join(dict.fromkeys(chars))


def parse_regex(pattern: str) -> list[tuple[str, int, int]]:
    """Parse the small regex subset synthschema generates from.

    Supported: literals, escaped literals, ``\\d``, ``\\w``, character classes such as
    ``[A-Z]`` or ``[A-Za-z0-9_]``, and the quantifiers ``{n}`` and ``{m,n}``.
    """
    tokens: list[tuple[str, int, int]] = []
    pos = 0
    pattern = pattern.strip("^$")
    while pos < len(pattern):
        m = _TOKEN_RE.match(pattern, pos)
        if m is None:  # pragma: no cover - the token regex matches any character
            raise ValueError(f"Cannot parse regex {pattern!r} at {pos}")
        text = m.group(0)
        if m.group(1) is not None:
            chars = _expand_class(m.group(1))
        elif text == r"\d":
            chars = _CLASS_CHARS["0-9"]
        elif text == r"\w":
            chars = _CLASS_CHARS["A-Z"] + _CLASS_CHARS["a-z"] + _CLASS_CHARS["0-9"] + "_"
        elif m.group(2) is not None:
            chars = m.group(2)
        else:
            chars = m.group(3)
        pos = m.end()
        lo = hi = 1
        q = _QUANT_RE.match(pattern, pos)
        if q:
            lo = int(q.group(1))
            hi = int(q.group(2)) if q.group(2) else lo
            pos = q.end()
        tokens.append((chars, lo, hi))
    return tokens


def _capacity(tokens: list[tuple[str, int, int]]) -> float:
    cap = 1.0
    for chars, _lo, hi in tokens:
        cap *= float(len(chars)) ** hi
        if cap > 1e18:
            return cap
    return cap


def _widen(tokens: list[tuple[str, int, int]], needed: int) -> list[tuple[str, int, int]]:
    """Grow the last variable token until the pattern can hold ``needed`` distinct values."""
    tokens = list(tokens)
    variable = [i for i, (c, _, _) in enumerate(tokens) if len(c) > 1]
    if not variable:
        tokens.append((_CLASS_CHARS["0-9"], 1, 1))
        variable = [len(tokens) - 1]
    i = variable[-1]
    while _capacity(tokens) < needed:
        chars, lo, hi = tokens[i]
        tokens[i] = (chars, lo + 1, hi + 1)
    return tokens


def _generate_tokens(tokens: list[tuple[str, int, int]], n: int, rng: np.random.Generator) -> np.ndarray:
    if n == 0:
        return np.array([], dtype=object)
    if all(lo == hi for _, lo, hi in tokens):
        parts = []
        for chars, k, _ in tokens:
            if k == 0:
                continue
            pool = np.array(list(chars))
            parts.append(pool[rng.integers(0, len(pool), size=(n, k))])
        if not parts:
            return np.full(n, "", dtype=object)
        mat = np.ascontiguousarray(np.concatenate(parts, axis=1))
        width = mat.shape[1]
        return mat.view(f"<U{width}").ravel().astype(object)
    out = []
    for _ in range(n):
        pieces = []
        for chars, lo, hi in tokens:
            k = int(rng.integers(lo, hi + 1))
            pieces.append("".join(chars[j] for j in rng.integers(0, len(chars), size=k)))
        out.append("".join(pieces))
    return np.array(out, dtype=object)


def generate_from_regex(
    pattern: str, n: int, rng: np.random.Generator, unique: bool = True
) -> np.ndarray:
    """Generate ``n`` strings that match ``pattern`` (unique by default)."""
    tokens = parse_regex(pattern)
    if not unique:
        return _generate_tokens(tokens, n, rng)
    if _capacity(tokens) < 2 * n:
        tokens = _widen(tokens, 2 * n)
    values = pd.unique(_generate_tokens(tokens, n, rng))
    for _ in range(50):
        if len(values) >= n:
            break
        extra = _generate_tokens(tokens, 2 * (n - len(values)) + 10, rng)
        values = pd.unique(np.concatenate([values, extra]))
    if len(values) < n:  # pragma: no cover - only with tiny pattern spaces
        raise ValueError(f"Could not generate {n} unique values for pattern {pattern!r}")
    return np.asarray(values[:n], dtype=object)


def infer_regex(values: Iterable, max_samples: int = 2000) -> str | None:
    """Learn a regex that describes a set of identifier strings, e.g. ``SKU-[0-9]{5}``."""
    vals = [str(v) for v in values]
    if not vals:
        return None
    if len(vals) > max_samples:
        idx = np.random.default_rng(0).choice(len(vals), max_samples, replace=False)
        vals = [vals[i] for i in idx]
    lengths = {len(v) for v in vals}
    if len(lengths) == 1:
        length = lengths.pop()
        position_chars = [{v[i] for v in vals} for i in range(length)]
        # A run of digit positions is a number: if any digit in the run varies, the whole
        # run is variable (so zero-padded counters such as 00042 become [0-9]{5}).
        numeric_run = [False] * length
        i = 0
        while i < length:
            if all(c.isdigit() for c in position_chars[i]):
                j = i
                while j < length and all(c.isdigit() for c in position_chars[j]):
                    j += 1
                if any(len(position_chars[k]) > 1 for k in range(i, j)):
                    for k in range(i, j):
                        numeric_run[k] = True
                i = j
            else:
                i += 1
        tokens: list[str] = []
        for i in range(length):
            chars = position_chars[i]
            if numeric_run[i]:
                tokens.append("[0-9]")
            elif len(chars) == 1:
                tokens.append(re.escape(next(iter(chars))))
            elif all(c.isdigit() for c in chars):
                tokens.append("[0-9]")
            elif all(c.isupper() and c.isalpha() for c in chars):
                tokens.append("[A-Z]")
            elif all(c.islower() and c.isalpha() for c in chars):
                tokens.append("[a-z]")
            elif all(c.isalnum() for c in chars):
                tokens.append("[A-Za-z0-9]")
            else:
                return None
        return _compress(tokens)
    prefix = _common_prefix(vals)
    rests = [v[len(prefix):] for v in vals]
    width = max(len(r) for r in rests)
    if all(r.isdigit() for r in rests if r):
        return re.escape(prefix) + f"[0-9]{{{width}}}"
    if all(r.isalnum() for r in rests if r):
        return re.escape(prefix) + f"[A-Za-z0-9]{{{width}}}"
    return None


def _compress(tokens: list[str]) -> str:
    out: list[str] = []
    i = 0
    while i < len(tokens):
        j = i
        while j + 1 < len(tokens) and tokens[j + 1] == tokens[i]:
            j += 1
        run = j - i + 1
        tok = tokens[i]
        if tok.startswith("[") and run > 1:
            out.append(f"{tok}{{{run}}}")
        else:
            out.append(tok * run)
        i = j + 1
    return "".join(out)


def _common_prefix(vals: list[str]) -> str:
    prefix = vals[0]
    for v in vals[1:]:
        while not v.startswith(prefix):
            prefix = prefix[:-1]
            if not prefix:
                return ""
    # Do not swallow a varying digit run into the prefix.
    while prefix and prefix[-1].isdigit():
        prefix = prefix[:-1]
    return prefix


# ---------------------------------------------------------------------- id generator


class IdGenerator:
    """Generates key values that look like the real ones but never copy them."""

    def __init__(self, column: Column | None = None) -> None:
        self.column = column or Column("id")
        self.kind = "int"
        self.start = 1
        self.pattern: str | None = self.column.regex_format
        self.unique_ratio = 1.0
        self.null_rate = 0.0

    def fit(self, series: pd.Series) -> IdGenerator:
        nonnull = series.dropna()
        self.null_rate = float(series.isna().mean()) if len(series) else 0.0
        self.unique_ratio = float(nonnull.nunique() / len(nonnull)) if len(nonnull) else 1.0
        if self.column.representation == "int" or (
            pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series)
        ):
            self.kind = "int"
            self.start = int(nonnull.min()) if len(nonnull) else 1
            return self
        if self.pattern is None and len(nonnull):
            sample = nonnull.astype(str)
            if sample.str.match(r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$").all():
                self.kind = "uuid"
                return self
            self.pattern = infer_regex(sample)
        self.kind = "regex" if self.pattern else "uuid"
        return self

    def generate(
        self, n: int, rng: np.random.Generator, unique: bool | None = None, allow_nulls: bool = True
    ) -> np.ndarray:
        if unique is None:
            unique = self.unique_ratio >= 0.999
        k = n if unique else max(1, int(round(self.unique_ratio * n)))
        if self.kind == "int":
            pool = np.arange(self.start, self.start + k, dtype=np.int64)
        elif self.kind == "uuid":
            raw = rng.integers(0, 16, size=(k, 32))
            hexchars = np.array(list("0123456789abcdef"))[raw]
            strings = np.ascontiguousarray(hexchars).view("<U32").ravel()
            pool = np.array(
                [f"{s[:8]}-{s[8:12]}-4{s[13:16]}-{s[16:20]}-{s[20:]}" for s in strings], dtype=object
            )
        else:
            pool = generate_from_regex(self.pattern, k, rng, unique=True)
        values = pool if unique else pool[rng.integers(0, len(pool), size=n)]
        if allow_nulls and self.null_rate > 0:
            values = values.astype(object)
            values[rng.random(n) < self.null_rate] = None
        return values


# ---------------------------------------------------------------------- PII and text

_FIRST = (
    "James Mary Robert Patricia John Jennifer Michael Linda David Elizabeth William Barbara "
    "Richard Susan Joseph Jessica Thomas Sarah Charles Karen Daniel Lisa Matthew Nancy Anthony "
    "Sandra Mark Ashley Priya Arjun Wei Mei Omar Fatima Lucas Sofia Mateo Aisha Kenji Yuki"
).split()
_LAST = (
    "Smith Johnson Williams Brown Jones Garcia Miller Davis Rodriguez Martinez Hernandez Lopez "
    "Gonzalez Wilson Anderson Thomas Taylor Moore Jackson Martin Lee Perez Thompson White Harris "
    "Sanchez Clark Ramirez Lewis Robinson Patel Kumar Chen Wang Nguyen Kim Ali Khan Silva Rossi"
).split()
_STREETS = "Main Oak Pine Maple Cedar Elm Washington Lake Hill Park River Sunset Highland".split()
_SUFFIX = "St Ave Rd Blvd Ln Dr Way Ct".split()
_CITIES = (
    "Springfield Riverton Fairview Georgetown Franklin Greenville Bristol Clinton Madison "
    "Salem Arlington Ashland Dover Hudson Kingston"
).split()
_COMPANY_A = "Blue North Bright Prime Summit Silver Global Apex Pioneer Vertex Harbor Cedar".split()
_COMPANY_B = "Systems Labs Partners Holdings Analytics Logistics Foods Retail Health Energy".split()
_WORDS = (
    "data quality order customer product service delivery price value report account team "
    "support request update review payment store region market growth plan system process "
    "issue result project network design contract offer return good fast late great poor "
    "excellent average simple clear helpful friendly slow easy new standard premium"
).split()

_FAKER_METHODS = {
    "email": "email",
    "phone_number": "phone_number",
    "name": "name",
    "first_name": "first_name",
    "last_name": "last_name",
    "address": "address",
    "street_address": "street_address",
    "ssn": "ssn",
    "credit_card_number": "credit_card_number",
    "ip_address": "ipv4_private",
    "url": "url",
    "company": "company",
    "postcode": "postcode",
}

_FAKER_MAX_DIRECT = 100_000


def _faker(locale: str | list[str] | None, rng: np.random.Generator):
    try:
        from faker import Faker
    except ImportError:
        return None
    fake = Faker(locale)
    fake.seed_instance(int(rng.integers(0, 2**31 - 1)))
    return fake


def _pick(pool: list[str], n: int, rng: np.random.Generator) -> np.ndarray:
    return np.array(pool, dtype=object)[rng.integers(0, len(pool), size=n)]


def _digits(n: int, k: int, rng: np.random.Generator) -> np.ndarray:
    mat = np.array(list("0123456789"))[rng.integers(0, 10, size=(n, k))]
    return np.ascontiguousarray(mat).view(f"<U{k}").ravel().astype(object)


def _builtin_pii(sdtype: str, n: int, rng: np.random.Generator) -> np.ndarray:
    first, last = _pick(_FIRST, n, rng), _pick(_LAST, n, rng)
    if sdtype == "first_name":
        return first
    if sdtype == "last_name":
        return last
    if sdtype == "name":
        return first + " " + last
    if sdtype == "email":
        num = _digits(n, 4, rng)
        return np.char.lower((first + "." + last + num).astype(str)).astype(object) + "@example.com"
    if sdtype == "phone_number":
        return "+1-555-" + _digits(n, 3, rng) + "-" + _digits(n, 4, rng)
    if sdtype in {"address", "street_address"}:
        street = (
            rng.integers(1, 9999, size=n).astype(str).astype(object)
            + " " + _pick(_STREETS, n, rng) + " " + _pick(_SUFFIX, n, rng)
        )
        if sdtype == "street_address":
            return street
        return street + ", " + _pick(_CITIES, n, rng) + " " + _digits(n, 5, rng)
    if sdtype == "ssn":  # 9xx area numbers are never issued
        return "9" + _digits(n, 2, rng) + "-" + _digits(n, 2, rng) + "-" + _digits(n, 4, rng)
    if sdtype == "credit_card_number":  # 4000 0000 prefix is a well-known test range
        return "40000000" + _digits(n, 8, rng)
    if sdtype == "ip_address":
        octets = rng.integers(0, 256, size=(n, 3)).astype(str)
        return np.array(["10." + ".".join(r) for r in octets], dtype=object)
    if sdtype == "url":
        return "https://www." + np.char.lower(_pick(_COMPANY_A, n, rng).astype(str)).astype(object) + _digits(n, 3, rng) + ".example.com"
    if sdtype == "company":
        return _pick(_COMPANY_A, n, rng) + " " + _pick(_COMPANY_B, n, rng)
    if sdtype == "postcode":
        return _digits(n, 5, rng)
    raise ValueError(f"No generator for sdtype {sdtype!r}")


def generate_pii(
    sdtype: str, n: int, rng: np.random.Generator, locale: str | list[str] | None = None
) -> np.ndarray:
    """Generate ``n`` fake values of a PII type. Uses Faker when installed."""
    if n == 0:
        return np.array([], dtype=object)
    fake = _faker(locale, rng)
    if fake is None:
        return _builtin_pii(sdtype, n, rng)
    method = getattr(fake, _FAKER_METHODS[sdtype])
    size = min(n, _FAKER_MAX_DIRECT)
    values = np.array([method() for _ in range(size)], dtype=object)
    if sdtype == "address":
        values = np.array([v.replace("\n", ", ") for v in values], dtype=object)
    if size < n:
        values = values[rng.integers(0, size, size=n)]
    return values


class PiiGenerator:
    def __init__(self, sdtype: str, locale: str | list[str] | None = None) -> None:
        self.sdtype = sdtype
        self.locale = locale
        self.null_rate = 0.0

    def fit(self, series: pd.Series) -> PiiGenerator:
        self.null_rate = float(series.isna().mean()) if len(series) else 0.0
        return self

    def generate(self, n: int, rng: np.random.Generator) -> np.ndarray:
        values = generate_pii(self.sdtype, n, rng, self.locale).astype(object)
        if self.null_rate > 0:
            values[rng.random(n) < self.null_rate] = None
        return values


class TextGenerator:
    """Generates placeholder free text with a word-count profile like the real column."""

    def __init__(self, locale: str | list[str] | None = None) -> None:
        self.locale = locale
        self.mean_words = 8.0
        self.std_words = 3.0
        self.null_rate = 0.0

    def fit(self, series: pd.Series) -> TextGenerator:
        self.null_rate = float(series.isna().mean()) if len(series) else 0.0
        counts = series.dropna().astype(str).str.split().str.len()
        if len(counts):
            self.mean_words = float(max(counts.mean(), 1))
            self.std_words = float(counts.std() if len(counts) > 1 else 0) or 1.0
        return self

    def generate(self, n: int, rng: np.random.Generator) -> np.ndarray:
        lengths = np.clip(np.round(rng.normal(self.mean_words, self.std_words, n)), 1, 200).astype(int)
        fake = _faker(self.locale, rng)
        if fake is not None and n <= _FAKER_MAX_DIRECT:
            values = np.array(
                [fake.sentence(nb_words=int(k), variable_nb_words=False) for k in lengths], dtype=object
            )
        else:
            words = np.array(_WORDS, dtype=object)
            values = np.array(
                [" ".join(words[rng.integers(0, len(words), size=k)]).capitalize() + "." for k in lengths],
                dtype=object,
            )
        if self.null_rate > 0:
            values[rng.random(n) < self.null_rate] = None
        return values
