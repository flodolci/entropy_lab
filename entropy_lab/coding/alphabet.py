"""
Core discrete-alphabet data structures for entropy_lab.

Design choices
--------------
- AlphabetDistribution is the *main* data class:
    - stores symbols + probability vector (aligned and validated)
    - provides basic "distribution-space" transforms (rank, mass queries, delta-subset)
    - stays independent from measures (entropy/KL/...) to avoid dependency tangles

- Measures live in entropy_lab/measures/ as standalone functions.
    - Optionally, we provide thin convencience methods on the class that call measures
      via *local imports* to avoid circular imports. 
"""

from __future__ import annotations

from dataclasses import dataclass 
from typing import Dict, Iterable, List, Sequence, Tuple, Optional, List, Mapping

import numpy as np
import numpy.typing as npt

# ---- Type aliases ----

FloatArray = npt.NDArray[np.float64]


# ---- Results/records ----

@dataclass(frozen=True)
class DeltaSubsetResult:
    """
    Result of computing a δ-sufficient (1-δ mass) subset of an alphabet.

    Attributes
    ----------
    kept:
        The renormalized distribution restricted to the kept symbols.
    dropped_symbols:
        Symbols removed from the alphabet to achieve the desired risk level.
    kept_probability:
        Total probability mass of the kept symbols under the *original* distribution.
    dropped_probability:
        Total probability mass of the dropped symbols under the *original* distribution.
    threshold:
        The target mass to keep, equal to 1 - δ.
    """
    kept: "AlphabetDistribution"
    dropped_symbols: Tuple[str, ...]
    kept_probability: float 
    dropped_probability: float 
    threshold: float # 1-delta 


# ---- Main data class ----

@dataclass(frozen=True)
class AlphabetDistribution:
    """
    A discrete alphabet with an associated probability distribution.

    Parameters
    ----------
    symbols:
        Tuple of symbol labels (e.g., characters, tokens, bin labels).
        Symbols are required to be unique
    p:
        Probability vector aligned with `symbols`. 
        Stored as float64, 1-D, finite, non-negative, and (within tolerance) sums to 1.

    Notes
    -----
    The class assumes `symbols[i]` occurs with probability `p[i]`.
    Use `from_probs` or `from_counts` to validate and (optionally) normalize.
    """

    symbols: Tuple[str, ...]
    p: FloatArray 

    # ---- Constructors ----

    @staticmethod
    def from_probs(
        symbols: Sequence[str], 
        p: npt.ArrayLike, 
        *, 
        normalize: bool = True,
        atol: float = 1e-12
    ) -> "AlphabetDistribution": 
        """
        Build an AlphabetDistribution from explicit probabilities.

        Parameters
        ----------
        symbols:
            Symbol labels.
        p:
            Non-negative weights or probabilities aligned with `symbols`.
        normalize:
            If True, renormalize so that probabilities sum to 1.
            If False, require `p.sum() == 1` within atol
        atol:
            Absolute tolerance for sum-to-1 validation when normalize=False.

        Returns
        -------
        AlphabetDistribution
            A validated (and optionally normalized) distribution.

        Raises
        ------
        ValueError
            If symbols are empty, duplicated, length mismatch, p invalid, etc.

        """
        symbols_t = tuple(symbols)
        p_arr = np.asarray(p, dtype=np.float64)

        # 1-D vector shape validation
        if p_arr.ndim != 1:
            raise ValueError(f"p must be 1-D (got shape {p_arr.shape}).")

        # length/emptiness validation
        if len(symbols_t) == 0:
            raise ValueError("alphabet cannot be empty.")
        if len(symbols_t) != p_arr.shape[0]:
            raise ValueError("symbols and p must have the same length.")

        # unique symbols enforcement
        if len(set(symbols_t)) != len(symbols_t):
            raise ValueError("symbols must be unique (duplicates found).")
        
        # numerical conditions validation
        if not np.all(np.isfinite(p_arr)):
            raise ValueError("Probabilities must be finite (no NaN/inf).")
        if np.any(p_arr < 0):
            raise ValueError("Probabilities must be non-negative.")

        s = float(p_arr.sum())
        if s <= 0:
            raise ValueError("At least one probability/weight must be > 0.")
        
        if normalize:
            p_arr = p_arr / s
        else:
            # require p sums to 1 within tolerance
            if not np.isclose(s, 1.0, atol=atol):
                raise ValueError(f"Probabilities must sum to 1 (got {s}).")
        
        # ensure contiguous for speed in numeric ops
        p_arr = np.ascontiguousarray(p_arr, dtype=np.float64)

        return AlphabetDistribution(symbols=symbols_t, p=p_arr)
    
    @staticmethod
    def from_counts(counts: Mapping[str, int]) -> "AlphabetDistribution":
        """
        Build an AlphabetDistribution from integer symbol counts.

        Parameters
        ----------
        counts:
            Mapping symbol -> count (non-negative). At least one count must be > 0.
            Order of symbols follows mapping iteration order

        Returns
        -------
        AlphabetDistribution
            Normalized probabilities proportional to counts.

        Raises
        ------
        ValueError
            If `counts` is empty, contains negative values, or all counts are zero.
        """
        if not counts:
            raise ValueError("counts cannot be empty.")
        
        symbols = tuple(counts.keys())
        c = np.array([counts[s] for s in symbols], dtype=np.float64)

        if np.any(c < 0):
            raise ValueError("Counts must be non-negative.")
        if float(c.sum()) <= 0:
            raise ValueError("At least one count must be > 0.")
        
        return AlphabetDistribution.from_probs(symbols, c, normalize=True)

    @staticmethod
    def from_samples(symbols: Sequence[str]) -> "AlphabetDistribution":
        """
        Convenience constructor: build an empirical distribution from a sequence of symbols.
        Useful when you already have discrete tokens (e.g., words/characters).
        """
        if not symbols:
            raise ValueError("symbols cannot be empty.")
        
        counts: Dict[str, int] = {}
        for s in symbols:
            counts[s] = counts.get(s, 0) + 1
        return AlphabetDistribution.from_counts(counts)

    # ---- Basic helpers ----

    def as_dict(self) -> Dict[str, float]:
        """
        Convert the distribution to a dict mapping symbol -> probability.
        """
        return {s: float(pi) for s, pi in zip(self.symbols, self.p)}
    
    def index(self) -> Dict[str, int]:
        """
        Return a symbol -> index mapping.
        """
        return {s: i for i, s in enumerate(self.symbols)}
    
    def ranked(self) -> "AlphabetDistribution":
        """
        Return a copy sorted by decreasing probability.

        Notes
        -----
        Uses a stable sort, so ties preserve the original input order.
        """
        order = np.argsort(-self.p, kind="mergesort")
        return AlphabetDistribution(
            symbols=tuple(self.symbols[i] for i in order),
            p=self.p[order].copy(),
        )
    
    def support(self, *, tol: float = 0.0) -> Tuple[str, ...]:
        """
        Returns the symbols in the support set (p > tol).

        Parameters
        ----------
        tol:
            Threshold for "non-zero". Use small tol to drop tiny numeric noise.
        """
        if tol < 0:
            raise ValueError("tol must be >= 0.")
        return tuple(s for s, pi in zip(self.symbols, self.p) if float(pi) > tol)
    
    def trim_zeros(self, *, tol: float = 0.0) -> "AlphabetDistribution":
        """
        Return a new distribution with symbols with p <= tol removed and normalized
        """
        keep_idx = [i for i, pi in enumerate(self.p) if float(pi) > tol]
        if not keep_idx:
            raise ValueError("All probabilities are <= tol; cannot trim to empty alphabet.")
        symbols = tuple(self.symbols[i] for i in keep_idx)
        p = self.p[keep_idx].copy()
        return AlphabetDistribution.from_probs(symbols, p, normalize=True)
    
    def mass_of(self, subset_symbols: Iterable[str]) -> float:
        """
        Compute total probability mass of a subset of symbols.

        Parameters
        ----------
        subset_symbols:
            Iterable of symbols to include.

        Returns
        -------
        float
            Sum of probabilities of symbols that are present in this distribution.
            Symbols not in the alphabet are ignored.
        """
        idx = self.index()
        total = 0.0
        for s in subset_symbols:
            i = idx.get(s)
            if i is not None:
                total += float(self.p[i])
        return total
    
    # ---- delta-sufficient subset
    
    def delta_subset(self, delta: float) -> DeltaSubsetResult:
        """
        Compute a δ-sufficient subset by dropping low-probability symbols.

        Interpretation
        --------------
        δ is the allowed "risk" (probability mass you are willing to lose).
        This returns the *smallest* set of most-likely symbols whose cumulative
        probability mass is at least 1 - δ, plus a renormalized distribution on
        that kept set.

        Parameters
        ----------
        delta:
            Risk budget in [0, 1]. A larger δ means you're willing to drop more.

        Returns
        -------
        DeltaSubsetResult
            Contains:
            - `kept`: renormalized distribution on the kept symbols,
            - `dropped_symbols`: symbols excluded,
            - kept/dropped probability masses under the original distribution,
            - `threshold` = 1 - δ.

        Raises
        ------
        ValueError
            If delta is outside [0, 1].

        Notes
        -----
        Edge cases:
        - δ = 0 keeps enough symbols to reach probability mass 1 (typically all).
        - δ = 1 sets threshold to 0; the current implementation will still keep
          at least one symbol because it takes the first symbol reaching the
          threshold with `searchsorted(...)+1`.
        """
        if not (0.0 <= delta <= 1.0):
            raise ValueError("delta must be in [0, 1].")

        ranked = self.ranked()
        threshold = 1.0 - delta
        cum = np.cumsum(ranked.p)

        # searchsorted gives first index where cum[idx] >= threshold
        # +1 so that k is a count (and ensures k >= 1)
        k = int(np.searchsorted(cum, threshold, side="left")) + 1
        k = min(k, len(ranked.symbols))  # can't exceed alphabet size

        kept_symbols = ranked.symbols[:k]
        kept_p = ranked.p[:k].copy()

        kept_prob = float(cum[k - 1])  # k >= 1 by design
        dropped_symbols = tuple(ranked.symbols[k:])

        # Protect against tiny negative values due to floating-point arithmetic
        dropped_prob = max(0.0, 1.0 - kept_prob)

        kept_dist = AlphabetDistribution.from_probs(kept_symbols, kept_p, normalize=True)

        return DeltaSubsetResult(
            kept=kept_dist,
            dropped_symbols=dropped_symbols,
            kept_probability=kept_prob,
            dropped_probability=dropped_prob,
            threshold=threshold,
        )
    
    # ---- convencience wrappers for measures ----
    
    def entropy(self, base: float = 2.0) -> float:
        """Convenience: return Shannon entropy of this distribution"""
        from entropy_lab.measures.entropy import compute_entropy
        return float(compute_entropy(self, base=base))
    
    def kl_divergence(self, other: "AlphabetDistribution", base: float = 2.0) -> float:
        """Convenience: KL(this || other)"""
        from entropy_lab.measures.divergences import compute_kl_divergence
        return float(compute_kl_divergence(self, other, base=base))



