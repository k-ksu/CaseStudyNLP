"""
Utility functions for analysing subword tokenization quality on
code-switched / mixed-language text.

This module provides metrics (fertility, OOV rate, sequence-length
inflation, vocabulary coverage), convenience wrappers that aggregate
those metrics, visualisation helpers, and lightweight per-token
language detection via *langid*.

All ``tokenizer`` parameters expect HuggingFace
``transformers.PreTrainedTokenizer`` (or ``PreTrainedTokenizerFast``)
objects.
"""

from __future__ import annotations

from collections import defaultdict
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
)

import langid
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast

# ---------------------------------------------------------------------------
# Global style configuration
# ---------------------------------------------------------------------------
plt.style.use("seaborn-v0_8-whitegrid")

COLOR_PALETTE: List[str] = sns.color_palette(
    "husl", n_colors=10
).as_hex()  # type: ignore[assignment]

sns.set_palette(COLOR_PALETTE)

# Type alias accepted for the tokenizer argument throughout this module.
Tokenizer = PreTrainedTokenizer | PreTrainedTokenizerFast


# ===================================================================
# 1. Fertility
# ===================================================================

def compute_fertility(
    tokenizer: Tokenizer,
    texts: Sequence[str],
    word_tokenizer: Optional[Callable[[str], List[str]]] = None,
) -> Dict[str, Any]:
    """Compute the average number of subword tokens per whitespace word.

    *Fertility* is a standard metric that captures how aggressively a
    tokenizer fragments the input.  A fertility of 1.0 means every word
    maps to exactly one subword token; higher values indicate more
    splitting.

    Parameters
    ----------
    tokenizer : Tokenizer
        A HuggingFace ``PreTrainedTokenizer`` or
        ``PreTrainedTokenizerFast``.
    texts : Sequence[str]
        A sequence of sentences / utterances to analyse.
    word_tokenizer : callable, optional
        A function ``str -> list[str]`` used to split a sentence into
        words.  When *None* (the default), plain whitespace splitting is
        used.

    Returns
    -------
    dict
        ``{"per_text": list[float], "mean": float}`` where each entry
        in *per_text* is the fertility for the corresponding input text.
    """
    split_fn: Callable[[str], List[str]] = (
        word_tokenizer if word_tokenizer is not None else str.split
    )

    per_text: List[float] = []

    for text in texts:
        words = split_fn(text)
        if len(words) == 0:
            per_text.append(0.0)
            continue

        token_ids = tokenizer.encode(text, add_special_tokens=False)
        fertility = len(token_ids) / len(words)
        per_text.append(fertility)

    mean_fertility = float(np.mean(per_text)) if per_text else 0.0

    return {"per_text": per_text, "mean": mean_fertility}


# ===================================================================
# 2. OOV (unknown-token) rate
# ===================================================================

def compute_oov_rate(
    tokenizer: Tokenizer,
    texts: Sequence[str],
) -> Dict[str, Any]:
    """Compute the percentage of ``[UNK]`` tokens produced by the tokenizer.

    This metric is only meaningful for tokenizers that actually have an
    ``unk_token`` (e.g. WordPiece).  For BPE/Unigram tokenizers that
    never produce ``[UNK]``, the rate will be 0.

    Parameters
    ----------
    tokenizer : Tokenizer
        A HuggingFace tokenizer.
    texts : Sequence[str]
        Sentences to analyse.

    Returns
    -------
    dict
        ``{"per_text": list[float], "mean": float}`` where each value
        is the OOV rate expressed as a **percentage** (0-100).
    """
    unk_id: Optional[int] = getattr(tokenizer, "unk_token_id", None)

    per_text: List[float] = []

    for text in texts:
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        if len(token_ids) == 0:
            per_text.append(0.0)
            continue

        if unk_id is not None:
            n_unk = sum(1 for tid in token_ids if tid == unk_id)
            oov_rate = (n_unk / len(token_ids)) * 100.0
        else:
            oov_rate = 0.0

        per_text.append(oov_rate)

    mean_oov = float(np.mean(per_text)) if per_text else 0.0

    return {"per_text": per_text, "mean": mean_oov}


# ===================================================================
# 3. Sequence-length inflation
# ===================================================================

def compute_sequence_length_inflation(
    tokenizer: Tokenizer,
    texts: Sequence[str],
) -> Dict[str, Any]:
    """Ratio of subword-token count to whitespace-word count per text.

    A value close to 1.0 indicates the tokenizer preserves the original
    sequence length; values >> 1 indicate significant inflation (more
    tokens than words), which can hurt models with limited context
    windows.

    Parameters
    ----------
    tokenizer : Tokenizer
        A HuggingFace tokenizer.
    texts : Sequence[str]
        Sentences to analyse.

    Returns
    -------
    dict
        ``{"per_text": list[float], "mean": float}``
    """
    per_text: List[float] = []

    for text in texts:
        words = text.split()
        if len(words) == 0:
            per_text.append(0.0)
            continue

        token_ids = tokenizer.encode(text, add_special_tokens=False)
        ratio = len(token_ids) / len(words)
        per_text.append(ratio)

    mean_ratio = float(np.mean(per_text)) if per_text else 0.0

    return {"per_text": per_text, "mean": mean_ratio}


# ===================================================================
# 4. Vocabulary coverage
# ===================================================================

def compute_vocabulary_coverage(
    tokenizer: Tokenizer,
    texts: Sequence[str],
) -> Dict[str, Any]:
    """Percentage of whitespace words tokenized into a *single* subword token.

    High coverage means most words in the corpus are already present in
    the tokenizer\'s vocabulary and do not need to be split.

    Parameters
    ----------
    tokenizer : Tokenizer
        A HuggingFace tokenizer.
    texts : Sequence[str]
        Sentences to analyse.

    Returns
    -------
    dict
        ``{"coverage_pct": float, "total_words": int,
        "single_token_words": int}``
    """
    total_words: int = 0
    single_token_words: int = 0

    for text in texts:
        for word in text.split():
            total_words += 1
            token_ids = tokenizer.encode(word, add_special_tokens=False)
            if len(token_ids) == 1:
                single_token_words += 1

    coverage = (single_token_words / total_words * 100.0) if total_words else 0.0

    return {
        "coverage_pct": coverage,
        "total_words": total_words,
        "single_token_words": single_token_words,
    }


# ===================================================================
# 5. Aggregate analysis
# ===================================================================

def tokenize_and_analyze(
    tokenizer: Tokenizer,
    texts: Sequence[str],
    tokenizer_name: str = "",
) -> Dict[str, Any]:
    """Run **all** metrics and return a flat summary dictionary.

    This is the main convenience entry-point: pass a tokenizer and a
    list of texts and get back a single dict suitable for appending to a
    ``pandas.DataFrame``.

    Parameters
    ----------
    tokenizer : Tokenizer
        A HuggingFace tokenizer.
    texts : Sequence[str]
        Sentences to analyse.
    tokenizer_name : str, optional
        A human-readable label for this tokenizer (e.g.
        ``"bert-base-multilingual-cased"``).

    Returns
    -------
    dict
        Keys: ``tokenizer_name``, ``mean_fertility``,
        ``mean_oov_rate``, ``mean_seq_inflation``,
        ``vocab_coverage``.
    """
    fertility = compute_fertility(tokenizer, texts)
    oov = compute_oov_rate(tokenizer, texts)
    inflation = compute_sequence_length_inflation(tokenizer, texts)
    coverage = compute_vocabulary_coverage(tokenizer, texts)

    return {
        "tokenizer_name": tokenizer_name,
        "mean_fertility": fertility["mean"],
        "mean_oov_rate": oov["mean"],
        "mean_seq_inflation": inflation["mean"],
        "vocab_coverage": coverage["coverage_pct"],
    }


# ===================================================================
# 6. Fertility bar chart
# ===================================================================

def plot_fertility_comparison(
    results_df: pd.DataFrame,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Bar chart comparing **fertility** across tokenizers.

    Parameters
    ----------
    results_df : pd.DataFrame
        Must contain columns ``tokenizer_name`` and ``mean_fertility``.
    save_path : str, optional
        If given, the figure is saved to this path (PNG/PDF/SVG).

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    bars = ax.bar(
        results_df["tokenizer_name"],
        results_df["mean_fertility"],
        color=COLOR_PALETTE[: len(results_df)],
        edgecolor="white",
        linewidth=0.8,
    )

    # Value labels on top of each bar.
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.02,
            f"{height:.2f}",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )

    ax.set_xlabel("Tokenizer", fontsize=13)
    ax.set_ylabel("Mean Fertility (subwords / word)", fontsize=13)
    ax.set_title(
        "Subword Fertility Comparison across Tokenizers",
        fontsize=15,
        fontweight="bold",
    )
    ax.tick_params(axis="x", rotation=25, labelsize=11)
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


# ===================================================================
# 7. Sequence-length inflation bar chart
# ===================================================================

def plot_sequence_length_comparison(
    results_df: pd.DataFrame,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Bar chart comparing **sequence-length inflation** across tokenizers.

    Parameters
    ----------
    results_df : pd.DataFrame
        Must contain columns ``tokenizer_name`` and
        ``mean_seq_inflation``.
    save_path : str, optional
        If given, the figure is saved to this path.

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    bars = ax.bar(
        results_df["tokenizer_name"],
        results_df["mean_seq_inflation"],
        color=COLOR_PALETTE[: len(results_df)],
        edgecolor="white",
        linewidth=0.8,
    )

    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.02,
            f"{height:.2f}",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )

    # Reference line at ratio = 1 (no inflation).
    ax.axhline(y=1.0, color="grey", linestyle="--", linewidth=1, label="No inflation")

    ax.set_xlabel("Tokenizer", fontsize=13)
    ax.set_ylabel("Mean Sequence-Length Inflation", fontsize=13)
    ax.set_title(
        "Sequence-Length Inflation Comparison across Tokenizers",
        fontsize=15,
        fontweight="bold",
    )
    ax.tick_params(axis="x", rotation=25, labelsize=11)
    ax.legend(fontsize=11)
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


# ===================================================================
# 8. Token-count-per-sentence histogram
# ===================================================================

def plot_token_distribution(
    tokenizer: Tokenizer,
    texts: Sequence[str],
    tokenizer_name: str = "",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Histogram of the number of tokens produced per sentence.

    Parameters
    ----------
    tokenizer : Tokenizer
        A HuggingFace tokenizer.
    texts : Sequence[str]
        Sentences to analyse.
    tokenizer_name : str, optional
        Label for the plot title.
    save_path : str, optional
        If given, the figure is saved to this path.

    Returns
    -------
    matplotlib.figure.Figure
    """
    token_counts: List[int] = [
        len(tokenizer.encode(text, add_special_tokens=False))
        for text in texts
    ]

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.hist(
        token_counts,
        bins="auto",
        color=COLOR_PALETTE[0],
        edgecolor="white",
        linewidth=0.8,
        alpha=0.85,
    )

    mean_count = float(np.mean(token_counts))
    ax.axvline(
        x=mean_count,
        color=COLOR_PALETTE[3],
        linestyle="--",
        linewidth=1.5,
        label=f"Mean = {mean_count:.1f}",
    )

    title_suffix = f" ({tokenizer_name})" if tokenizer_name else ""
    ax.set_xlabel("Tokens per Sentence", fontsize=13)
    ax.set_ylabel("Frequency", fontsize=13)
    ax.set_title(
        f"Token-Count Distribution{title_suffix}",
        fontsize=15,
        fontweight="bold",
    )
    ax.legend(fontsize=11)
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


# ===================================================================
# 9. Per-token language detection
# ===================================================================

def detect_language_per_token(
    text: str,
) -> List[Tuple[str, str]]:
    """Detect the language of each whitespace-delimited word using *langid*.

    This is a lightweight heuristic -- single-word language identification
    is inherently noisy, but it gives a useful first approximation for
    code-switched text analysis.

    Parameters
    ----------
    text : str
        A single sentence / utterance, potentially code-switched.

    Returns
    -------
    list[tuple[str, str]]
        A list of ``(word, detected_language_code)`` pairs.  Language
        codes follow ISO 639-1 (e.g. ``"en"``, ``"hi"``, ``"es"``).
    """
    results: List[Tuple[str, str]] = []

    for word in text.split():
        lang, _confidence = langid.classify(word)
        results.append((word, lang))

    return results


# ===================================================================
# 10. Per-language fertility
# ===================================================================

def per_language_fertility(
    tokenizer: Tokenizer,
    texts_with_lang_labels: Sequence[Sequence[Tuple[str, str]]],
) -> Dict[str, float]:
    """Compute fertility **per language group**.

    Parameters
    ----------
    tokenizer : Tokenizer
        A HuggingFace tokenizer.
    texts_with_lang_labels : sequence of sequences of (word, lang) tuples
        Typically produced by calling :func:`detect_language_per_token`
        on every sentence in your corpus.  Each inner sequence
        represents one sentence and contains ``(word, lang)`` pairs.

    Returns
    -------
    dict[str, float]
        Mapping from language code to mean fertility for words assigned
        to that language.

    Example
    -------
    >>> labeled = [detect_language_per_token(t) for t in texts]
    >>> per_language_fertility(tokenizer, labeled)
    {"en": 1.12, "hi": 2.34, "es": 1.56}
    """
    lang_subword_counts: Dict[str, List[int]] = defaultdict(list)

    for sentence in texts_with_lang_labels:
        for word, lang in sentence:
            token_ids = tokenizer.encode(word, add_special_tokens=False)
            lang_subword_counts[lang].append(len(token_ids))

    fertility_per_lang: Dict[str, float] = {
        lang: float(np.mean(counts))
        for lang, counts in sorted(lang_subword_counts.items())
    }

    return fertility_per_lang
