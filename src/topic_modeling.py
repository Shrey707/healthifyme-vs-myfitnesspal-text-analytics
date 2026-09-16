"""
src/topic_modeling.py
---------------------
Reusable functions for embedding generation and BERTopic training.
- Embeddings are cached to .npy; never recomputed on reruns.
- BERTopic models are saved to models/ for reuse by later notebooks.
"""

from __future__ import annotations

import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from bertopic import BERTopic
from hdbscan import HDBSCAN
from sentence_transformers import SentenceTransformer
from umap import UMAP


# ── Embedding ─────────────────────────────────────────────────────────────────

def get_or_create_embeddings(
    texts: list[str],
    cache_path: str,
    model_name: str = "all-MiniLM-L6-v2",
    batch_size: int = 256,
) -> np.ndarray:
    """
    Load embeddings from cache_path if it exists, otherwise compute and save.

    Parameters
    ----------
    texts      : list of strings to embed
    cache_path : path to .npy file (created if absent)
    model_name : sentence-transformers model identifier
    batch_size : encoding batch size

    Returns
    -------
    np.ndarray of shape (n_docs, embedding_dim)
    """
    cache = Path(cache_path)
    if cache.exists():
        embeddings = np.load(cache)
        print(f"  [cache hit]  Loaded embeddings from {cache}  shape={embeddings.shape}")
        return embeddings

    print(f"  [cache miss] Computing embeddings with {model_name} for {len(texts):,} docs ...")
    model = SentenceTransformer(model_name)
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
    )
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache, embeddings)
    print(f"  [saved]      {cache}  shape={embeddings.shape}")
    return embeddings


# ── BERTopic ──────────────────────────────────────────────────────────────────

def build_bertopic_model(
    n_neighbors: int = 15,
    n_components: int = 5,
    min_cluster_size: int = 50,
    min_samples: int = 10,
    nr_topics: str | int = "auto",
    random_state: int = 42,
) -> BERTopic:
    """
    Construct a BERTopic instance with explicit UMAP and HDBSCAN settings
    so the run is reproducible.
    """
    umap_model = UMAP(
        n_neighbors=n_neighbors,
        n_components=n_components,
        min_dist=0.0,
        metric="cosine",
        random_state=random_state,
    )
    hdbscan_model = HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric="euclidean",
        cluster_selection_method="eom",
        prediction_data=True,
    )
    topic_model = BERTopic(
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        nr_topics=nr_topics,
        top_n_words=10,
        verbose=True,
    )
    return topic_model


def run_bertopic(
    texts: list[str],
    embeddings: np.ndarray,
    app_name: str,
    models_dir: str = "models",
    **bertopic_kwargs,
) -> tuple[BERTopic, list[int], list[float]]:
    """
    Fit BERTopic on texts+embeddings, save model, return (model, topics, probs).

    If a saved model already exists at models/{app_name}_bertopic/ the model
    is loaded and transform() is skipped in favour of the cached topic assignments
    stored alongside it. This avoids UMAP non-determinism on repeated runs.
    """
    # pickle serialization → single .pkl file; topic arrays alongside it
    model_file  = Path(models_dir) / f"{app_name}_bertopic.pkl"
    topics_file = Path(models_dir) / f"{app_name}_topics.npy"
    probs_file  = Path(models_dir) / f"{app_name}_probs.npy"

    if model_file.exists() and topics_file.exists():
        print(f"  [cache hit]  Loading BERTopic model from {model_file}")
        topic_model = BERTopic.load(str(model_file))
        topics = np.load(topics_file).tolist()
        probs  = np.load(probs_file).tolist()
        return topic_model, topics, probs

    print(f"  [training]   Fitting BERTopic for {app_name} on {len(texts):,} docs ...")
    topic_model = build_bertopic_model(**bertopic_kwargs)
    topics, probs = topic_model.fit_transform(texts, embeddings)

    # Save model + topic assignments
    Path(models_dir).mkdir(parents=True, exist_ok=True)
    topic_model.save(str(model_file), serialization="pickle", save_ctfidf=True)
    np.save(topics_file, np.array(topics))
    np.save(probs_file,  np.array(probs))
    print(f"  [saved]      Model → {model_file}")

    return topic_model, topics, probs


# ── Reporting ─────────────────────────────────────────────────────────────────

def topic_report(
    topic_model: BERTopic,
    topics: list[int],
    app_name: str,
    top_n_words: int = 10,
) -> pd.DataFrame:
    """
    Print a formatted topic report and return a summary DataFrame.

    Columns: topic_id, doc_count, top_words
    """
    topic_counts = pd.Series(topics).value_counts().sort_index()
    n_outlier    = topic_counts.get(-1, 0)
    n_real       = len(topics) - n_outlier
    n_topics     = len([t for t in topic_counts.index if t != -1])

    print(f"\n{'='*65}")
    print(f"  BERTopic Report: {app_name}")
    print(f"{'='*65}")
    print(f"  Total documents      : {len(topics):,}")
    print(f"  Real topics found    : {n_topics}")
    print(f"  Docs in real topics  : {n_real:,}  ({n_real/len(topics)*100:.1f}%)")
    print(f"  Docs in outlier (-1) : {n_outlier:,}  ({n_outlier/len(topics)*100:.1f}%)")

    print(f"\n  {'Topic':>6}  {'Docs':>7}  {'%':>5}  Top words")
    print(f"  {'─'*6}  {'─'*7}  {'─'*5}  {'─'*45}")

    rows = []
    topic_info = topic_model.get_topic_info()
    for _, row in topic_info.sort_values("Count", ascending=False).iterrows():
        tid   = row["Topic"]
        count = topic_counts.get(tid, 0)
        pct   = count / len(topics) * 100
        words = topic_model.get_topic(tid)
        if words:
            top_words = ", ".join(w for w, _ in words[:top_n_words])
        else:
            top_words = "(outlier — no representative words)"
        label = f"{'[outlier]':>6}" if tid == -1 else f"{tid:>6}"
        print(f"  {label}  {count:>7,}  {pct:>4.1f}%  {top_words}")
        rows.append({
            "topic_id":  tid,
            "doc_count": count,
            "pct":       round(pct, 2),
            "top_words": top_words,
        })

    return pd.DataFrame(rows)
