import numpy as np
import pandas as pd

from entropy_lab.measures.shannon import shannon_information
from entropy_lab.text.models import ReferenceLanguageModel
from entropy_lab.text.preprocessing import preprocess_text

from typing import Tuple, List
from collections import Counter

def score_text_with_reference(
        text: str,
        ref_model: ReferenceLanguageModel,
        base: float = 2.0
) -> Tuple[List[str], np.ndarray, pd.DataFrame]:
    """
    For each token in 'text':
        - compute p_ref(token)
        - compute surprisal h(token) = -log_base(p_ref(token))

    Returns
    -------
    tokens: list[str]
    surprisals: np.ndarray
    token_df: pd.DataFrame with columns [token, p_ref, surprisal_bits]
    """
    tokens = preprocess_text(text)
    rows = []
    surprisals = []
    for tok in tokens:
        p_ref = ref_model.p(tok)
        h = shannon_information(p_ref, base=base)
        surprisals.append(h)
        rows.append({
            "token": tok,
            "p_ref": p_ref,
            "surprisal_bits": h
        })
    token_df = pd.DataFrame(rows)
    return tokens, np.array(surprisals, dtype=float), token_df

def summarize_surprisal(tokens: List[str], surprisals: np.ndarray) -> pd.Series:
    """
    Summary stats to compare texts.
    """
    if len(tokens) == 0:
        raise ValueError("No tokens to summarize.")
    n_tokens = len(tokens)
    n_unique = len(set(tokens))
    type_token_ratio = n_unique / n_tokens
    summary = pd.Series({
        "n_tokens": n_tokens,
        "n_unique": n_unique,
        "type_token_ratio": type_token_ratio,
        "mean_surprisal_bits": float(np.mean(surprisals)),
        "median_surprisal_bits": float(np.median(surprisals)),
        "std_surprisal_bits": float(np.std(surprisals)),
        "frac_surprisal_gt_8": float(np.mean(surprisals > 8)),
        "frac_surprisal_gt_10": float(np.mean(surprisals > 10))
    })
    return summary

def build_entropy_table(tokens: List[str], base: float = 2.0) -> Tuple[pd.DataFrame, float]:
    """
    Build a table:
        token | count | p_i | h(p_i) | p_i * h(p_i)

    using the empirical distribution of the given token list.

    Returns
    -------
    entropy_table: pd.DataFrame
    H: float (empirical Shannon entropy in bits/token)
    """
    counts = Counter(tokens)
    N = sum(counts.values())
    rows = []
    for token, c in counts.items():
        p_i = c / N
        h_i = shannon_information(p_i, base=base)
        contrib = p_i * h_i 
        rows.append({
            "token": token,
            "count": c,
            "p_i": p_i,
            "h(p_i)_bits": h_i,
            "p_i*h(p_i)": contrib
        })
    df = pd.DataFrame(rows)
    df = df.sort_values("count", ascending=False).reset_index(drop=True)
    H = float(df["p_i*h(p_i)"].sum())
    return df, H