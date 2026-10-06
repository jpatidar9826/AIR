"""Load and standardise the public Kaggle datasets.

Column names on Kaggle can change between versions, so instead of hard-coding
them we map several known aliases to a standard schema:

    screening data       -> resume, job_description, job_role, label (0/1)
    resume category data -> resume, category
"""
from __future__ import annotations

import os
import re
import shutil
from pathlib import Path

import pandas as pd

import config

ALIASES = {
    "resume": ["resume", "resume_str", "resume_text", "cv", "cv_text"],
    "job_description": ["job_description", "jd", "job_desc", "description", "job_description_text"],
    "job_role": ["job_roles", "job_role", "role", "job_title", "position", "title"],
    "label": ["best_match", "label", "shortlisted", "recruiter_decision", "selected", "hired", "match"],
    "category": ["category", "resume_category"],
}

POSITIVE_LABELS = {"1", "yes", "y", "true", "hire", "hired", "selected", "shortlisted", "match", "best match"}
NEGATIVE_LABELS = {"0", "no", "n", "false", "reject", "rejected", "not selected", "no match"}


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [re.sub(r"[^a-z0-9]+", "_", str(c).strip().lower()).strip("_") for c in df.columns]
    return df


def find_column(df: pd.DataFrame, key: str) -> str | None:
    for alias in ALIASES[key]:
        if alias in df.columns:
            return alias
    return None


def _read_csv(path: Path) -> pd.DataFrame:
    for enc in ("utf-8", "latin-1"):
        try:
            return pd.read_csv(path, encoding=enc)
        except UnicodeDecodeError:
            continue
    raise ValueError(f"Could not decode {path}")


def _csv_files() -> list[Path]:
    return sorted(p for p in config.RAW_DIR.rglob("*.csv"))


def download_from_kaggle(which: str = "all") -> list[Path]:
    """Download datasets with kagglehub (needs internet + Kaggle credentials).

    If the AI Lab has no internet, download the CSVs manually from the URLs in
    config.KAGGLE_DATASETS and place them in data/raw/.
    """
    try:
        import kagglehub
    except ImportError:
        print("kagglehub not installed. Run `pip install kagglehub` or download manually.")
        return []

    copied: list[Path] = []
    keys = config.KAGGLE_DATASETS.keys() if which == "all" else [which]
    for key in keys:
        meta = config.KAGGLE_DATASETS[key]
        try:
            src = Path(kagglehub.dataset_download(meta["slug"]))
        except Exception as exc:  # network / auth errors
            print(f"[{key}] download failed: {exc}\n  -> download manually from {meta['url']}")
            continue
        dest = config.RAW_DIR / key
        dest.mkdir(parents=True, exist_ok=True)
        for csv in src.rglob("*.csv"):
            target = dest / csv.name
            shutil.copy(csv, target)
            copied.append(target)
            print(f"[{key}] saved {target.relative_to(config.ROOT)}")
    return copied


def _pick_file(env_var: str, required: list[str], forbidden: list[str] | None = None) -> Path:
    explicit = os.getenv(env_var, "").strip()
    if explicit:
        return Path(explicit)
    for path in _csv_files():
        try:
            head = normalize_columns(pd.read_csv(path, nrows=5, encoding_errors="ignore"))
        except Exception:
            continue
        if all(find_column(head, k) for k in required) and not any(
            find_column(head, k) for k in (forbidden or [])
        ):
            return path
    raise FileNotFoundError(
        f"No CSV in {config.RAW_DIR} has columns for {required}. "
        f"Run `python run_pipeline.py --download` or place the Kaggle CSV in data/raw/ "
        f"(or set {env_var} in .env)."
    )


def to_binary_label(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        return (series.astype(float) > 0).astype(int)
    s = series.astype(str).str.strip().str.lower()
    mapped = s.map(lambda v: 1 if v in POSITIVE_LABELS else (0 if v in NEGATIVE_LABELS else None))
    return mapped


def load_screening_data() -> pd.DataFrame:
    path = _pick_file("SCREENING_FILE", ["resume", "job_description", "label"])
    raw = normalize_columns(_read_csv(path))
    out = pd.DataFrame({
        "resume": raw[find_column(raw, "resume")].astype(str),
        "job_description": raw[find_column(raw, "job_description")].astype(str),
        "label": to_binary_label(raw[find_column(raw, "label")]),
    })
    role_col = find_column(raw, "job_role")
    out["job_role"] = raw[role_col].astype(str) if role_col else "Unknown"
    # keep sensitive attributes ONLY for the fairness audit (never as features)
    for col in config.SENSITIVE_COLUMNS:
        if col in raw.columns:
            out[f"sensitive_{col}"] = raw[col]

    before = len(out)
    out = out.dropna(subset=["resume", "job_description", "label"])
    out = out[(out.resume.str.len() > 20) & (out.job_description.str.len() > 20)]
    out = out.drop_duplicates(subset=["resume", "job_description"]).reset_index(drop=True)
    out["label"] = out["label"].astype(int)
    out.attrs["source_file"] = str(path)
    out.attrs["rows_dropped"] = before - len(out)
    return out


def load_resume_category_data() -> pd.DataFrame:
    path = _pick_file("RESUME_CATEGORY_FILE", ["resume", "category"], forbidden=["job_description"])
    raw = normalize_columns(_read_csv(path))
    out = pd.DataFrame({
        "resume": raw[find_column(raw, "resume")].astype(str),
        "category": raw[find_column(raw, "category")].astype(str).str.strip(),
    }).dropna()
    out = out[out.resume.str.len() > 50].drop_duplicates().reset_index(drop=True)
    out.attrs["source_file"] = str(path)
    return out
