#!/usr/bin/env python3
"""Reference outputs for the text and datetime preprocessors.

TabPFN v3.5's estimator runs two frame-level transformers before it
validates its input: `DateTransformer`, which expands each point in time
into calendar features, then `TextTransformer`, which expands each text
column into 30 LSA components by way of skrub's `StringEncoder`. The R port
lives in `R/prep-datetime.R`, `R/prep-text.R` and `R/prep-frame.R`; this
script records what the reference produces so that
`tests/testthat/test-prep-text.R`, `test-prep-datetime.R` and
`test-prep-frame.R` can check it with no Python at test time.

Needs tabpfn 9.0.0 and skrub, which only `.venvs/ref35` has -- the
tabpfn 8.2.0 in `.venvs/ref` predates both transformers:

    .venvs/ref35/bin/python inst/parity/textdate_reference.py \\
        --out inst/parity/textdate/textdate.safetensors

What is recorded, and why each piece is there:

* **The whole estimator sequence on one mixed frame** -- a number, a
  duration, a low-cardinality string, a naive timestamp with fractional
  seconds, a timezone-aware one crossing a DST change, a date-only column,
  and a text column -- through `DateTransformer` then `TextTransformer`, at
  fit and on unseen rows. That checks the layout (kept columns, then dates,
  then text), the names, durations converted in place, and that a
  low-cardinality string is *not* treated as text.
* **`StringEncoder` alone on three corpora**, from well-conditioned to
  very flat. On a flat spectrum the randomized SVD's trailing components
  are decided by `random_state = 0` rather than by the data -- two seeds
  disagree by nearly one unit -- so these are the fixtures that fail if the
  port's NumPy stream is a single draw out of step.
* **A float64 twin of each**: scikit-learn's own algorithm run at float64
  on the same float32-rounded tf-idf matrix. The reference computes in
  float32 and R cannot, so the port is graded twice: tightly against the
  twin, which proves the algorithm is the same, and loosely against the
  real output, where the remaining gap is scikit-learn's own float32
  rounding.
* **A small-vocabulary corpus**, for skrub's branch that skips the SVD and
  keeps the leading tf-idf columns.

Arrays go to a gzipped safetensors file as float64; strings and scalars go
to a sibling JSON -- see `transforms_reference.py` for why both halves are
needed and why the tensor file is gzipped.
"""

from __future__ import annotations

import argparse
import gzip
import json
import shutil
import string
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from safetensors.torch import save_file
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from skrub import StringEncoder

import sklearn
import skrub
import tabpfn
from tabpfn.preprocessing.datetimes import DateTransformer
from tabpfn.preprocessing.text import TextTransformer

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument("--out", required=True)
args = parser.parse_args()

tensors: dict[str, np.ndarray] = {}
scalars: dict[str, object] = {}


def corpus(n_docs: int, vocab: int, seed: int) -> tuple[list, list]:
    rng = np.random.default_rng(seed)
    words = [
        "".join(rng.choice(list(string.ascii_lowercase), size=rng.integers(3, 10)))
        for _ in range(vocab)
    ]
    docs = [" ".join(rng.choice(words, size=rng.integers(2, 8))) for _ in range(n_docs)]
    return docs, words


def twin(docs: list) -> tuple[np.ndarray, np.ndarray, float]:
    """scikit-learn's LSA at float64, from the float32-rounded matrix."""
    filled = [d if d is not None else "" for d in docs]
    tv = TfidfVectorizer(ngram_range=(3, 4), analyzer="char_wb")
    x32 = tv.fit_transform(filled).astype("float32")
    x64 = x32.astype("float64")
    svd = TruncatedSVD(n_components=30, random_state=0).fit(x64)
    res = np.asarray(x64 @ svd.components_.T)
    factor = float(np.sqrt(np.nansum(np.nanvar(res, ddof=0, axis=0))))
    return res / factor, svd.components_, factor


# ---------------------------------------------------------------------------
# StringEncoder alone
# ---------------------------------------------------------------------------

for name, (n_docs, vocab, seed) in {
    "wellcond": (300, 40, 0),
    "flat": (80, 300, 1),
    "veryflat": (50, 500, 2),
}.items():
    docs, words = corpus(n_docs, vocab, seed)
    docs[3] = None
    test = [" ".join(words[:3]), "completely unseen zzqx", None, docs[0]]
    enc = StringEncoder(n_components=30, random_state=0)
    real = enc.fit_transform(pd.Series(docs, dtype="string", name="t"))
    real_test = enc.transform(pd.Series(test, dtype="string", name="t"))
    tw, comp, factor = twin(docs)
    scalars[f"text_{name}_docs"] = docs
    scalars[f"text_{name}_test_docs"] = test
    scalars[f"text_{name}_names"] = [str(c) for c in real.columns]
    scalars[f"text_{name}_scaling_twin"] = factor
    tensors[f"text_{name}_real"] = real.to_numpy("float64")
    tensors[f"text_{name}_test_real"] = real_test.to_numpy("float64")
    tensors[f"text_{name}_twin"] = tw
    tensors[f"text_{name}_components_twin"] = comp

# Too few n-grams for an SVD: skrub keeps the leading tf-idf columns.
small = ["a", "b", "a b", "b a", "a a", None, "b b a", "ab", "ba"] * 4
enc = StringEncoder(n_components=30, random_state=0)
out = enc.fit_transform(pd.Series(small, dtype="string", name="s"))
scalars["text_small_docs"] = small
scalars["text_small_names"] = [str(c) for c in out.columns]
scalars["text_small_has_svd"] = hasattr(enc, "tsvd_")
tensors["text_small_real"] = out.to_numpy("float64")

# ---------------------------------------------------------------------------
# The estimator's sequence on one mixed frame
# ---------------------------------------------------------------------------

rng = np.random.default_rng(11)
n_fit, n_new = 60, 10
n = n_fit + n_new
base = pd.Timestamp("2023-03-01")
# Minutes into the year, plus a fractional second on some rows so the
# `second` feature has something to floor.
offsets = np.sort(rng.integers(0, 60 * 24 * 400, size=n)) * 60.0
offsets[::7] += 0.75
naive = base + pd.to_timedelta(offsets, unit="s")
naive = pd.Series(naive).astype("datetime64[ns]")
naive.iloc[5] = pd.NaT
aware = naive.dt.tz_localize("UTC").dt.tz_convert("America/New_York")
dates = pd.Series(pd.to_datetime("2020-02-27") + pd.to_timedelta(rng.integers(0, 1500, size=n), unit="D"))
dates.iloc[9] = pd.NaT
docs, words = corpus(n, 250, 5)
docs[4] = None
low = pd.Series(rng.choice(["red", "green", "blue", None], size=n), dtype="string")

frame = pd.DataFrame({
    "a": rng.normal(size=n),
    "dur": pd.to_timedelta(rng.uniform(1, 5000, size=n).round(3), unit="s"),
    "low": low,
    "ts": naive,
    "ny": aware,
    "d": dates,
    "note": pd.Series(docs, dtype="string"),
})

fit_rows, new_rows = frame.iloc[:n_fit], frame.iloc[n_fit:].reset_index(drop=True)
dt = DateTransformer(transform_dates=True)
tt = TextTransformer(transform_text=True)
fit_out = tt.fit_transform(dt.fit_transform(fit_rows))
new_out = tt.transform(dt.transform(new_rows))

numeric_cols = [c for c in fit_out.columns if c != "low"]
scalars["frame_output_names"] = [str(c) for c in fit_out.columns]
scalars["frame_numeric_names"] = [str(c) for c in numeric_cols]
scalars["frame_text_positions"] = tt.expanded_indices
scalars["frame_date_positions"] = dt.expanded_indices
# The raw inputs, in forms R can rebuild exactly: epoch seconds for the
# timestamps (the fractional ones are exact in binary), days for the dates,
# seconds for the durations.
scalars["frame_in_a"] = frame["a"].tolist()
scalars["frame_in_dur_seconds"] = frame["dur"].dt.total_seconds().tolist()
scalars["frame_in_low"] = [None if pd.isna(v) else str(v) for v in frame["low"]]
scalars["frame_in_ts_epoch"] = [
    None if pd.isna(v) else (v - pd.Timestamp("1970-01-01")).total_seconds() for v in naive
]
scalars["frame_in_d_days"] = [
    None if pd.isna(v) else (v - pd.Timestamp("1970-01-01")).days for v in dates
]
scalars["frame_in_note"] = docs
scalars["frame_n_fit"] = n_fit
tensors["frame_fit"] = fit_out[numeric_cols].astype("float64").to_numpy()
tensors["frame_new"] = new_out[numeric_cols].astype("float64").to_numpy()

# The frame's text block at float64, fit and new rows, for the same
# two-sided grading the standalone corpora get.
fit_docs = [d if d is not None else "" for d in docs[:n_fit]]
new_docs = [d if d is not None else "" for d in docs[n_fit:]]
tv = TfidfVectorizer(ngram_range=(3, 4), analyzer="char_wb")
x_fit = tv.fit_transform(fit_docs).astype("float32").astype("float64")
x_new = tv.transform(new_docs).astype("float32").astype("float64")
svd = TruncatedSVD(n_components=30, random_state=0).fit(x_fit)
r_fit = np.asarray(x_fit @ svd.components_.T)
factor = np.sqrt(np.nansum(np.nanvar(r_fit, ddof=0, axis=0)))
tensors["frame_note_twin_fit"] = r_fit / factor
tensors["frame_note_twin_new"] = np.asarray(x_new @ svd.components_.T) / factor

# Detection rules on their own: which string columns count as text.
detect = pd.DataFrame({
    "text": pd.Series(docs, dtype="string"),
    "few": pd.Series(rng.choice(list("abcdefghij"), size=n), dtype="string"),
    "numbers_as_text": pd.Series([f"{v:.3f}" for v in rng.normal(size=n)], dtype="string"),
    # 30 distinct values exactly, one of them missing: *not* more than 30.
    "boundary": pd.Series(([f"v{i}" for i in range(29)] + [None]) * 3)[:n].astype("string"),
})
tt_detect = TextTransformer(transform_text=True).fit(detect)
scalars["detect_columns"] = list(detect.columns)
scalars["detect_values"] = {c: [None if pd.isna(v) else str(v) for v in detect[c]] for c in detect.columns}
scalars["detect_text_positions"] = tt_detect.expanded_indices

scalars["_versions"] = {
    "tabpfn": tabpfn.__version__, "sklearn": sklearn.__version__,
    "skrub": skrub.__version__, "numpy": np.__version__, "pandas": pd.__version__,
}

out_path = Path(args.out)
out_path.parent.mkdir(parents=True, exist_ok=True)
save_file(
    {k: torch.from_numpy(np.ascontiguousarray(np.asarray(v, dtype="float64")))
     for k, v in tensors.items()},
    str(out_path),
)
with open(out_path.with_suffix(".json"), "w") as fh:
    json.dump(scalars, fh, indent=1)
with open(out_path, "rb") as src, gzip.open(f"{out_path}.gz", "wb", compresslevel=9) as dst:
    shutil.copyfileobj(src, dst)
out_path.unlink()
print(f"[py] wrote {len(tensors)} arrays to {out_path}.gz and "
      f"{len(scalars)} fields to {out_path.with_suffix('.json')}")
