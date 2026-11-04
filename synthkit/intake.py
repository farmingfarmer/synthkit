"""Read whatever file the operator actually has.

The fitted path has only ever read CSV. An operator with an
.xlsx export or a .jsonl dump had to convert it first, and the
error they got when they did not was pandas' own - a parser
complaint about a binary file, which says nothing about what to
do next.

THE READER IS PART OF THE SUSPECT PATH, so this module keeps
the semantics the CSV path spent a long time getting right
rather than inventing new ones:

- EVERY COLUMN IS A STRING. The typing pass decides what a
  column is; letting the reader guess means a silent wrong type,
  which is the fault class this project has recorded four times.
- A BLANK FIELD IS "" AND NOT MISSING. `keep_default_na=False`
  is why; present-and-empty is a fact about a set column, and
  folding it into missing erased 36% of the medication-free
  visits on the real extract once already.
- NOTHING IS DROPPED OR RENAMED. A reader that quietly cleans
  is a reader that quietly changes the answer.

A format we cannot read is REFUSED BY NAME with the install
that would fix it. "Could not read as CSV" on a .xlsx is a
true sentence that helps nobody.
"""
from pathlib import Path

import pandas as pd

# Extensions we recognise, and what each needs. Kept as data so
# the refusal can name the install rather than the caller having
# to know which package backs which format.
FORMATS = {
    ".csv": (None, "comma-separated text"),
    ".tsv": (None, "tab-separated text"),
    ".txt": (None, "delimited text"),
    ".jsonl": (None, "JSON Lines, one record per line"),
    ".ndjson": (None, "newline-delimited JSON"),
    ".json": (None, "JSON records"),
    ".parquet": ("pyarrow", "Apache Parquet"),
    ".xlsx": ("openpyxl", "Excel workbook"),
    ".xlsm": ("openpyxl", "Excel workbook with macros"),
    ".xls": ("xlrd", "legacy Excel workbook"),
}


def _have(mod):
    try:
        __import__(mod)
        return True
    except ImportError:
        return False


def describe_support():
    """One line per format, saying whether it can be read now."""
    out = []
    for ext in sorted(FORMATS):
        need, what = FORMATS[ext]
        if need is None:
            out.append("{:<9} {:<34} yes".format(ext, what))
        elif _have(need):
            out.append("{:<9} {:<34} yes ({})".format(
                ext, what, need))
        else:
            out.append("{:<9} {:<34} NO - pip install {}".format(
                ext, what, need))
    return out


class IntakeError(Exception):
    """A refusal that names the file and what would fix it."""


def read_table(src, max_rows=None, sheet=None):
    """Read a tabular file as all-strings, blanks preserved.

    Returns (frame, note) where note says how it was read, so a
    run can print it. A reader that does not say what it did is
    the silent-wrong-type fault waiting to happen.
    """
    src = Path(src)
    if not src.exists():
        raise IntakeError("{} does not exist".format(src))
    ext = src.suffix.lower()

    if ext not in FORMATS:
        raise IntakeError(
            "{} is a {} file and this tool reads: {}. Convert "
            "it, or say which of those it really is by "
            "renaming.".format(
                src.name, ext or "(no extension)",
                ", ".join(sorted(FORMATS))))

    need, what = FORMATS[ext]
    if need and not _have(need):
        raise IntakeError(
            "{} is {} and reading it needs the `{}` package, "
            "which is not installed. Run: pip install {}"
            .format(src.name, what, need, need))

    try:
        if ext in (".csv", ".tsv", ".txt"):
            sep = "\t" if ext == ".tsv" else ","
            df = pd.read_csv(src, dtype=str,
                             keep_default_na=False,
                             encoding="utf-8-sig", sep=sep,
                             nrows=max_rows)
            note = "{} as delimited text".format(src.name)
        elif ext in (".jsonl", ".ndjson"):
            df = pd.read_json(src, lines=True, dtype=str,
                              nrows=max_rows)
            df = _as_strings(df)
            note = "{} as JSON Lines".format(src.name)
        elif ext == ".json":
            # A .json file is EITHER a list of records or one
            # record per line. Try the array form and fall back,
            # because both are called .json by the tools that
            # write them and guessing wrong is a crash an
            # operator cannot act on.
            try:
                df = pd.read_json(src, dtype=str)
                note = "{} as a JSON array of records".format(
                    src.name)
            except ValueError:
                df = pd.read_json(src, lines=True, dtype=str)
                note = ("{} as JSON Lines (the array form did "
                        "not parse)".format(src.name))
            df = _as_strings(df)
            if max_rows:
                df = df.head(int(max_rows))
        elif ext == ".parquet":
            df = pd.read_parquet(src)
            df = _as_strings(df)
            if max_rows:
                df = df.head(int(max_rows))
            note = "{} as Parquet".format(src.name)
        else:
            # A WORKBOOK HAS SHEETS AND A CSV DOES NOT. Reading
            # the first one silently is how an operator gets a
            # run over their notes tab. The sheet is named in
            # the note, and the others are named in the error
            # when the requested one is absent.
            book = pd.ExcelFile(src)
            names = list(book.sheet_names)
            if sheet is None:
                pick = names[0]
            elif sheet in names:
                pick = sheet
            else:
                raise IntakeError(
                    "{} has no sheet called {!r}. It has: {}"
                    .format(src.name, sheet, ", ".join(names)))
            # READ NATIVELY, NOT AS STRINGS. pandas' own
            # dtype=str conversion renders an integral float
            # 60.0 as "60" while the CSV path preserves
            # "60.0", so passing it here put a second renderer
            # in front of this module's and the two formats
            # disagreed on exactly those cells. Reading
            # natively leaves one renderer - the one below,
            # which decides integrality per COLUMN.
            df = book.parse(pick, nrows=max_rows)
            df = _as_strings(df)
            note = "{} sheet {!r}{}".format(
                src.name, pick,
                " (of {}: {})".format(len(names),
                                      ", ".join(names))
                if len(names) > 1 else "")
    except IntakeError:
        raise
    except Exception as exc:
        raise IntakeError(
            "could not read {} as {}: {}".format(
                src.name, what, exc))

    if df.shape[1] == 0:
        raise IntakeError(
            "{} parsed with no columns - it may be empty or in "
            "a shape this reader did not expect".format(
                src.name))
    return df, note


def _as_strings(df):
    """Every column a string, every absence an empty string.

    The CSV path gets this from `keep_default_na=False`; the
    other readers have no such flag, so a NaN would arrive where
    the CSV path delivers "". Two readers disagreeing about what
    an absent cell is would put the empty-versus-missing fault
    back exactly where it was fixed.
    """
    out = df.copy()
    for c in out.columns:
        s = out[c]
        if s.dtype == object:
            out[c] = s.where(s.notna(), "").astype(str)
        else:
            # INTEGRALITY IS A PROPERTY OF THE COLUMN, NOT THE
            # CELL, and deciding it per cell made the formats
            # disagree.
            #
            # A workbook stores numbers, not text, so a float
            # column arrives with no way to know whether the
            # source wrote 3 or 3.0. Rendering any integral
            # value without its tail fixed the integer column
            # and broke the float one: a normal draw that
            # happened to land on exactly 60.0 came back as
            # "60" from Excel against "60.0" from the CSV.
            # Caught by the check that every format must
            # produce the identical frame, which is the only
            # check here that could have caught it.
            #
            # So the whole column decides: all values integral
            # means an integer column and no tails; anything
            # fractional means a float column and every value
            # keeps its own rendering.
            vals = s.dropna()
            whole = bool(len(vals)) and all(
                _is_integral(v) for v in vals)
            out[c] = s.map(
                _render_int if whole else _render).astype(str)
        out[c] = out[c].where(df[c].notna(), "")
    out.columns = [str(c) for c in out.columns]
    return out


def _is_integral(v):
    import math
    if isinstance(v, bool):
        return False
    if isinstance(v, float):
        return not math.isnan(v) and v.is_integer()
    return isinstance(v, int)


def _render_int(v):
    """A value from a column whose values are all whole."""
    import math
    if v is None:
        return ""
    if isinstance(v, float):
        if math.isnan(v):
            return ""
        return str(int(v))
    return str(v)


def _render(v):
    """A value from a column that carries fractions."""
    import math
    if v is None:
        return ""
    if isinstance(v, float) and math.isnan(v):
        return ""
    return str(v)
