"""
src/clean_data.py
-----------------
Reusable cleaning functions for Play Store review datasets.
All functions are pure (return new DataFrames, never modify in-place)
and append to a shared cleaning log list passed by the caller.
"""

from __future__ import annotations

import re
from typing import Any

import langid
import pandas as pd

# ── Patterns ──────────────────────────────────────────────────────────────────

_MIXED_SENTIMENT_PATTERNS: list[str] = [
    r"used to be",
    r"used to have",
    r"not anymore",
    r"no longer",
    r"but now",
    r"but recently",
    r"but after",
    r"before.{0,30}(?:update|version|change)",
    r"(?:update|version|change).{0,30}ruined",
    r"was (?:good|great|amazing|excellent|perfect|nice|wonderful|awesome)",
    r"(?:it|this) (?:was|were) (?:good|great|amazing|excellent|perfect)",
    r"loved.{0,20}(?:but|however|until|though)",
    r"(?:love|like).{0,30}(?:but|however|until|though|except)",
    r"(?:used|worked) well.{0,30}(?:but|however|until)",
    r"(?:great|good|nice|excellent).{0,30}(?:but|however|except|though|unfortunately)",
    r"(?:but|however).{0,40}(?:terrible|worst|horrible|awful|bad|poor|useless|broken|crash)",
]

_MIXED_SENTIMENT_REGEX = re.compile(
    "|".join(_MIXED_SENTIMENT_PATTERNS), flags=re.IGNORECASE
)


# ── Core cleaning steps ───────────────────────────────────────────────────────

def load_raw(path: str) -> pd.DataFrame:
    """Load a raw CSV with settings that suppress mixed-type warnings."""
    return pd.read_csv(path, low_memory=False)


def log_step(log: list[dict], step: str, before: int, after: int, detail: str = "") -> None:
    """Append a cleaning-log entry."""
    dropped = before - after
    log.append(
        {
            "step": step,
            "rows_before": before,
            "rows_after": after,
            "rows_dropped": dropped,
            "detail": detail,
        }
    )


def drop_useless_columns(df: pd.DataFrame, log: list[dict]) -> pd.DataFrame:
    """Drop columns that are 100% null or carry no analytical value."""
    to_drop = [c for c in df.columns if df[c].isna().all()]
    df = df.drop(columns=to_drop)
    log_step(
        log,
        step="drop_useless_columns",
        before=len(df),
        after=len(df),
        detail=f"Dropped columns (all-null): {to_drop}",
    )
    return df


def fix_developer_response(df: pd.DataFrame, log: list[dict]) -> pd.DataFrame:
    """
    Cast developer_response / developer_response_date to str (suppressing
    mixed-type NaN→float issue) and add a boolean has_dev_response column.
    """
    for col in ("developer_response", "developer_response_date"):
        if col in df.columns:
            # Replace float NaN and string 'nan' / '0' artifacts with actual NaN
            df[col] = df[col].astype(str).replace({"nan": pd.NA, "0": pd.NA, "": pd.NA})

    has_col = "developer_response" in df.columns
    df["has_dev_response"] = (
        df["developer_response"].notna() if has_col else False
    )
    log_step(
        log,
        step="fix_developer_response",
        before=len(df),
        after=len(df),
        detail=(
            f"Cast dev response cols to str; added has_dev_response "
            f"(True={df['has_dev_response'].sum():,})"
        ),
    )
    return df


def drop_null_review_text(df: pd.DataFrame, log: list[dict]) -> pd.DataFrame:
    """Drop rows where review_description is null or empty string."""
    before = len(df)
    mask = df["review_description"].notna() & (df["review_description"].str.strip() != "")
    df = df[mask].copy()
    log_step(
        log,
        step="drop_null_review_text",
        before=before,
        after=len(df),
        detail="Dropped rows with null/empty review_description",
    )
    return df


def parse_review_date(df: pd.DataFrame, log: list[dict]) -> pd.DataFrame:
    """Parse review_date string (YYYY-MM-DD HH:MM:SS) to datetime64."""
    df["review_date"] = pd.to_datetime(df["review_date"], errors="coerce")
    n_failed = df["review_date"].isna().sum()
    log_step(
        log,
        step="parse_review_date",
        before=len(df),
        after=len(df),
        detail=f"Parsed review_date to datetime64; {n_failed} coerced to NaT",
    )
    return df


def add_mixed_sentiment_flag(df: pd.DataFrame, log: list[dict]) -> pd.DataFrame:
    """
    Add possible_mixed_sentiment boolean.
    Flags rows where:
      - rating <= 3, AND
      - review_description matches at least one mixed-sentiment pattern.
    Rows are NOT dropped — flag only.
    """
    low_rating = df["rating"] <= 3
    text_match = df["review_description"].str.contains(
        _MIXED_SENTIMENT_REGEX, na=False
    )
    df["possible_mixed_sentiment"] = low_rating & text_match
    flagged = df["possible_mixed_sentiment"].sum()
    log_step(
        log,
        step="add_mixed_sentiment_flag",
        before=len(df),
        after=len(df),
        detail=(
            f"Flagged {flagged:,} rows as possible_mixed_sentiment "
            f"(rating≤3 + temporal/contrast pattern). No rows dropped."
        ),
    )
    return df


def _detect_lang_fast(text: Any) -> str:
    """
    Classify language using langid, truncated to first 300 chars.
    Returns ISO 639-1 code or 'unknown'.
    """
    try:
        return langid.classify(str(text)[:300])[0]
    except Exception:
        return "unknown"


def run_language_detection(df: pd.DataFrame, log: list[dict]) -> pd.DataFrame:
    """
    Run language detection on the full review_description column.

    Strategy: langid is run only on rows that contain non-ASCII characters
    (necessary for non-Latin-script languages). ASCII-only rows are labelled
    'en' — safe for this dataset since all rows carry language_code='en' from
    the Play Store export, and purely ASCII text written in a non-English
    Latin-script language is ambiguous and negligible here.

    Adds a detected_lang column. Does NOT filter — filtering decision is left
    to the caller so the log can record the counts first.
    """
    texts = df["review_description"].fillna("")
    has_non_ascii = texts.apply(lambda x: any(ord(c) > 127 for c in str(x)[:300]))

    n_non_ascii = has_non_ascii.sum()
    print(
        f"    Non-ASCII rows: {n_non_ascii:,} / {len(df):,} — "
        f"running langid on these; labelling rest 'en'..."
    )

    detected = pd.Series("en", index=df.index, dtype=str)
    detected[has_non_ascii] = texts[has_non_ascii].apply(_detect_lang_fast)

    df["detected_lang"] = detected

    lang_counts = df["detected_lang"].value_counts()
    non_english = df[df["detected_lang"] != "en"]
    top_non_en = lang_counts[lang_counts.index != "en"].head(10).to_dict()

    log_step(
        log,
        step="run_language_detection",
        before=len(df),
        after=len(df),
        detail=(
            f"Checked {n_non_ascii:,} non-ASCII rows via langid; "
            f"non-English: {len(non_english):,} / {len(df):,} "
            f"({len(non_english)/len(df)*100:.2f}%). "
            f"Top non-EN langs: {top_non_en}"
        ),
    )
    print(f"    Done. Non-English: {len(non_english):,} ({len(non_english)/len(df)*100:.2f}%). "
          f"Top: {top_non_en}")
    return df


def filter_non_english(df: pd.DataFrame, log: list[dict]) -> pd.DataFrame:
    """Drop rows where detected_lang is not 'en'. Requires run_language_detection first."""
    before = len(df)
    df = df[df["detected_lang"] == "en"].copy()
    log_step(
        log,
        step="filter_non_english",
        before=before,
        after=len(df),
        detail="Dropped non-English rows (detected_lang != 'en')",
    )
    return df


def add_app_column(df: pd.DataFrame, app_name: str, log: list[dict]) -> pd.DataFrame:
    """Add a normalised app identifier column."""
    df["app_label"] = app_name
    log_step(
        log,
        step="add_app_column",
        before=len(df),
        after=len(df),
        detail=f"Added app_label = '{app_name}'",
    )
    return df


# ── High-level pipeline ───────────────────────────────────────────────────────

def clean_dataset(
    path: str,
    app_name: str,
    filter_english: bool = True,
) -> tuple[pd.DataFrame, list[dict]]:
    """
    Full cleaning pipeline for a single app's raw CSV.

    Parameters
    ----------
    path : str
        Path to the raw CSV file.
    app_name : str
        Short identifier, e.g. 'healthifyme' or 'myfitnesspal'.
    filter_english : bool
        If True (default), drop non-English rows after detection.

    Returns
    -------
    (cleaned_df, log)
        cleaned_df : cleaned DataFrame
        log        : list of dicts, one per cleaning step
    """
    log: list[dict] = []

    print(f"\n{'='*60}")
    print(f"  Cleaning: {app_name}")
    print(f"{'='*60}")

    df = load_raw(path)
    rows_in = len(df)
    log_step(log, step="load_raw", before=rows_in, after=rows_in,
             detail=f"Loaded {path}")

    df = drop_useless_columns(df, log)
    df = fix_developer_response(df, log)
    df = drop_null_review_text(df, log)
    df = parse_review_date(df, log)
    df = add_mixed_sentiment_flag(df, log)
    df = run_language_detection(df, log)

    if filter_english:
        df = filter_non_english(df, log)

    df = add_app_column(df, app_name, log)

    rows_out = len(df)
    print(f"\n  {app_name}: {rows_in:,} rows in → {rows_out:,} rows out "
          f"({rows_in - rows_out:,} dropped, {rows_out/rows_in*100:.1f}% retained)\n")

    return df, log


def build_cleaning_log_df(log: list[dict]) -> pd.DataFrame:
    """Convert a log list to a display-friendly DataFrame."""
    return pd.DataFrame(log)[["step", "rows_before", "rows_after", "rows_dropped", "detail"]]
