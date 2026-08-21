"""Compare a rebuilt table against a published one, cell by cell.

The published tables come out of a spreadsheet, which stores numbers and renders
them to a fixed width. A rebuilt value therefore rarely matches character for
character even when it is the same number. Two cells are equal when:

  * the strings are identical, or
  * both are empty / "NA", or
  * the reference is a bound like "<1e-300" and ours is below it, or
  * both parse as numbers and ours, rounded to the digits the reference shows,
    is the reference (allowing one unit in that last displayed digit, since the
    spreadsheet and Python break ties differently).

Anything else is a mismatch.
"""
import math
import re

_DIGITS = re.compile(r"[^0-9]")


def _significant_digits(s):
    mantissa = s.split("e")[0].split("E")[0]
    digits = _DIGITS.sub("", mantissa).lstrip("0")
    return max(len(digits), 1)


def cells_equal(ref, got):
    ref, got = ref.strip(), got.strip()
    if ref == got:
        return True
    if ref in ("", "NA") and got in ("", "NA"):
        return True
    if ref.startswith("<"):
        try:
            return float(got) < float(ref[1:])
        except ValueError:
            return False
    try:
        a, b = float(ref), float(got)
    except ValueError:
        return False
    if a == b:
        return True
    n = _significant_digits(ref)
    unit = 10 ** (math.floor(math.log10(abs(a))) - (n - 1)) if a else 10 ** -n
    return abs(a - b) <= unit


def compare(ref_df, got_df, label_col=None, max_examples=3):
    """Return a list of human-readable mismatch descriptions ([] if identical)."""
    problems = []
    if list(ref_df.columns) != list(got_df.columns):
        problems.append(f"columns differ\n      published: {list(ref_df.columns)}"
                        f"\n      rebuilt  : {list(got_df.columns)}")
        return problems
    if len(ref_df) != len(got_df):
        problems.append(f"{len(got_df)} rows rebuilt, published has {len(ref_df)}")
        return problems

    for col in ref_df.columns:
        rows = [i for i in range(len(ref_df))
                if not cells_equal(ref_df[col][i], got_df[col][i])]
        if not rows:
            continue
        examples = []
        for i in rows[:max_examples]:
            where = f"row {i + 1}"
            if label_col:
                where += f" ({ref_df[label_col][i]})"
            examples.append(f"        {where}: published={ref_df[col][i]!r} "
                            f"rebuilt={got_df[col][i]!r}")
        problems.append(f"column '{col}': {len(rows)} of {len(ref_df)} cells differ\n"
                        + "\n".join(examples))
    return problems
