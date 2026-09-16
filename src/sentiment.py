"""
src/sentiment.py
----------------
VADER and transformer sentiment scoring functions.
- VADER: fast, runs on full cleaned datasets
- Transformer: batched inference with progress reporting, runs on topic datasets
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


def _print(*args, **kwargs):
    """print() wrapper that always flushes — needed when running as a background task."""
    kwargs.setdefault("flush", True)
    print(*args, **kwargs)

# ── VADER ─────────────────────────────────────────────────────────────────────

_vader = SentimentIntensityAnalyzer()


def _vader_label(compound: float) -> str:
    if compound >= 0.05:
        return "positive"
    if compound <= -0.05:
        return "negative"
    return "neutral"


def run_vader(df: pd.DataFrame, text_col: str = "review_description") -> pd.DataFrame:
    """
    Add vader_compound, vader_label columns to a copy of df.
    Returns the augmented DataFrame.
    """
    scores = df[text_col].fillna("").apply(
        lambda t: _vader.polarity_scores(str(t))["compound"]
    )
    df = df.copy()
    df["vader_compound"] = scores.round(4)
    df["vader_label"]    = scores.apply(_vader_label)
    return df


# ── Transformer ───────────────────────────────────────────────────────────────

MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"

# Label map: the Cardiff model uses these raw labels
_LABEL_MAP = {
    "positive": "positive",
    "neutral":  "neutral",
    "negative": "negative",
    # fallback for older checkpoint label names
    "LABEL_0":  "negative",
    "LABEL_1":  "neutral",
    "LABEL_2":  "positive",
}


def run_transformer(
    df: pd.DataFrame,
    text_col: str = "review_description",
    batch_size: int = 64,
    max_length: int = 128,
    progress_every: int = 5_000,
    cache_path: str | None = None,
) -> pd.DataFrame:
    """
    Add roberta_label, roberta_score columns via batched pipeline inference.

    Parameters
    ----------
    batch_size     : docs per inference batch
    max_length     : truncation length in tokens (128 covers ~95% of reviews)
    progress_every : print a progress line every N rows
    cache_path     : if set, load from this .parquet if it exists, else save after inference

    Returns
    -------
    df copy with roberta_label (str) and roberta_score (float, winning class prob)
    """
    from transformers import pipeline

    if cache_path and Path(cache_path).exists():
        cached = pd.read_parquet(cache_path)
        _print(f"  [cache hit]  Loaded transformer scores from {cache_path}  ({len(cached):,} rows)")
        # merge on index — cached rows are in original order
        df = df.copy().reset_index(drop=True)
        df["roberta_label"] = cached["roberta_label"].values
        df["roberta_score"] = cached["roberta_score"].values
        return df

    _print(f"  [loading]    {MODEL_NAME}")
    pipe = pipeline(
        "text-classification",
        model=MODEL_NAME,
        tokenizer=MODEL_NAME,
        truncation=True,
        max_length=max_length,
        batch_size=batch_size,
        top_k=1,
    )

    texts  = df[text_col].fillna("").tolist()
    n      = len(texts)
    labels = []
    scores = []

    t_start = time.time()
    t_last  = t_start

    _print(f"  [inference]  {n:,} docs  |  batch_size={batch_size}  max_length={max_length}")

    for i in range(0, n, progress_every):
        chunk     = texts[i : i + progress_every]
        results   = pipe(chunk)
        for res in results:
            top = res[0] if isinstance(res, list) else res
            labels.append(_LABEL_MAP.get(top["label"], top["label"].lower()))
            scores.append(round(top["score"], 4))

        elapsed   = time.time() - t_start
        chunk_sec = time.time() - t_last
        done      = min(i + progress_every, n)
        rate      = progress_every / chunk_sec if chunk_sec > 0 else 0
        remaining = (n - done) / rate if rate > 0 else 0
        _print(
            f"  {done:>7,} / {n:,}  |  {rate:,.0f} docs/sec  |  "
            f"elapsed {elapsed:.0f}s  |  ETA {remaining:.0f}s"
        )
        t_last = time.time()

    df = df.copy().reset_index(drop=True)
    df["roberta_label"] = labels
    df["roberta_score"] = scores

    if cache_path:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        df[["roberta_label", "roberta_score"]].to_parquet(cache_path, index=False)
        _print(f"  [saved]      Transformer scores → {cache_path}")

    return df


# ── Disagreement analysis ─────────────────────────────────────────────────────

def disagreement_report(
    df: pd.DataFrame,
    flag_col: str = "possible_mixed_sentiment",
    vader_col: str = "vader_label",
    roberta_col: str = "roberta_label",
) -> None:
    """
    Print how VADER and transformer distribute sentiment labels on:
      (a) all rows in df
      (b) possible_mixed_sentiment == True rows
    """
    flagged = df[df[flag_col] == True]

    print(f"\n{'='*65}")
    print(f"  SENTIMENT DISAGREEMENT ANALYSIS")
    print(f"  Scope: {len(df):,} rows total  |  {len(flagged):,} flagged mixed-sentiment")
    print(f"{'='*65}")

    for label, subset in [("ALL rows", df), ("FLAGGED (possible_mixed_sentiment=True)", flagged)]:
        n = len(subset)
        if n == 0:
            print(f"\n  {label}: no rows")
            continue

        vader_dist   = subset[vader_col].value_counts()
        roberta_dist = subset[roberta_col].value_counts()

        print(f"\n  ── {label}  (n={n:,}) ──")
        print(f"  {'Label':<12}  {'VADER n':>8}  {'VADER %':>8}  {'RoBERTa n':>10}  {'RoBERTa %':>10}")
        print(f"  {'─'*12}  {'─'*8}  {'─'*8}  {'─'*10}  {'─'*10}")
        for lbl in ["positive", "neutral", "negative"]:
            vn = vader_dist.get(lbl, 0)
            rn = roberta_dist.get(lbl, 0)
            print(f"  {lbl:<12}  {vn:>8,}  {vn/n*100:>7.1f}%  {rn:>10,}  {rn/n*100:>9.1f}%")

        # agreement rate
        agree = (subset[vader_col] == subset[roberta_col]).sum()
        print(f"\n  Agreement rate: {agree:,} / {n:,} = {agree/n*100:.1f}%")

        # on flagged rows: show the specific disagreement pattern
        if "FLAGGED" in label and n > 0:
            cross = pd.crosstab(
                subset[vader_col], subset[roberta_col],
                rownames=["VADER"], colnames=["RoBERTa"]
            )
            print(f"\n  Confusion matrix (VADER rows × RoBERTa cols):")
            print(cross.to_string())

            # the key question: how many flagged rows does each model call positive/neutral
            # (i.e., miss the underlying negativity)?
            v_pos_neu = (subset[vader_col].isin(["positive","neutral"])).sum()
            r_pos_neu = (subset[roberta_col].isin(["positive","neutral"])).sum()
            print(f"\n  Flagged rows scored positive/neutral (i.e., missed as negative):")
            print(f"    VADER   : {v_pos_neu:,} / {n:,} = {v_pos_neu/n*100:.1f}%")
            print(f"    RoBERTa : {r_pos_neu:,} / {n:,} = {r_pos_neu/n*100:.1f}%")
            print(f"\n  Flagged rows scored negative by BOTH models:")
            both_neg = ((subset[vader_col]=="negative") & (subset[roberta_col]=="negative")).sum()
            print(f"    {both_neg:,} / {n:,} = {both_neg/n*100:.1f}%")
            print(f"\n  Flagged rows where models DISAGREE on positive vs negative:")
            v_pos_r_neg = ((subset[vader_col].isin(["positive","neutral"])) & (subset[roberta_col]=="negative")).sum()
            v_neg_r_pos = ((subset[vader_col]=="negative") & (subset[roberta_col].isin(["positive","neutral"]))).sum()
            print(f"    VADER pos/neu, RoBERTa neg: {v_pos_r_neg:,} ({v_pos_r_neg/n*100:.1f}%)")
            print(f"    VADER neg, RoBERTa pos/neu: {v_neg_r_pos:,} ({v_neg_r_pos/n*100:.1f}%)")
