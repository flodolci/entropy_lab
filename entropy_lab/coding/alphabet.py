from __future__ import annotations
from dataclasses import dataclass 
from typing import Dict, Iterable, List, Sequence, Tuple, Optional
import numpy as np
import numpy.typing as npt

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
    dropped_symbols: List[str]
    kept_probability: float 
    dropped_probability: float 
    threshold: float # 1-delta 

@dataclass(frozen=True)
class AlphabetDistribution:
    """
    A discrete alphabet with an associated probability distribution.

    Parameters
    ----------
    symbols:
        Tuple of symbol labels (e.g., characters).
    p:
        Probability vector aligned with `symbols`. Typically sums to 1.

    Notes
    -----
    The class assumes `symbols[i]` occurs with probability `p[i]`.
    Use `from_probs` or `from_counts` to validate and (optionally) normalize.
    """
    symbols: Tuple[str, ...]
    p: np.ndarray   # normalized 

    @staticmethod
    def from_probs(symbols: Sequence[str], p: npt.NDArray[np.floating], *, normalize: bool = True) -> "AlphabetDistribution": 
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
            If False, require `p.sum() == 1` (within numerical tolerance).

        Returns
        -------
        AlphabetDistribution
            A validated (and optionally normalized) distribution.

        Raises
        ------
        ValueError
            If lengths mismatch, probabilities are negative, all zeros,
            or (when normalize=False) probabilities do not sum to 1.
        """
        symbols = tuple(symbols)
        p = np.asarray(p, dtype=float)

        if len(symbols) != len(p):
            raise ValueError("symbols and p must have the same length")
        if np.any(p < 0):
            raise ValueError("Probabilities must be non-negative.")
        if np.all(p == 0):
            raise ValueError("At least one probability must be > 0.")
        
        if normalize:
            p = p / p.sum()
        else:
            s = p.sum()
            if not np.isclose(s, 1.0):
                raise ValueError(f"Probabilities must sum to 1 (got {s}).")
        return AlphabetDistribution(symbols=symbols, p=p)
    
    @staticmethod
    def from_counts(counts: Dict[str, int]) -> "AlphabetDistribution":
        """
        Build an AlphabetDistribution from integer symbol counts.

        Parameters
        ----------
        counts:
            Mapping symbol -> count (non-negative). At least one count must be > 0.

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
        symbols = list(counts.keys())
        c = np.array([counts[s] for s in symbols], dtype=float)
        if np.any(c < 0):
            raise ValueError("Counts must be non-negative.")
        if np.all(c == 0):
            raise ValueError("At least one count must be > 0.")
        return AlphabetDistribution.from_probs(symbols, c, normalize=True)
    
    def as_dict(self) -> Dict[str, float]:
        """
        Convert the distribution to a dict mapping symbol -> probability.
        """
        return {s: float(pi) for s, pi in zip(self.symbols, self.p)}
    
    def ranked(self) -> "AlphabetDistribution":
        """
        Return a copy sorted by decreasing probability.

        Notes
        -----
        Uses a stable sort, so ties preserve the original input order.
        """
        order = np.argsort(-self.p, kind="mergesort")
        symbols = tuple(self.symbols[i] for i in order)
        p = self.p[order]
        return AlphabetDistribution(symbols=symbols, p=p)
    
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
        subset = set(subset_symbols)
        return float(sum(self.p[i] for i, s in enumerate(self.symbols) if s in subset))
    
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
        if not(0.0 <= delta <= 1.0):
            raise ValueError("delta must be in [0, 1].")
        ranked = self.ranked()
        threshold = 1.0 - delta 
        cum = np.cumsum(ranked.p)
        k = int(np.searchsorted(cum, threshold, side="left")) + 1
        k = min(max(k, 0), len(ranked.symbols))
        kept_symbols = ranked.symbols[:k]
        kept_p = ranked.p[:k].copy()
        kept_prob = float(cum[k - 1]) if k > 0 else 0.0
        dropped_symbols = list(ranked.symbols[k:])
        dropped_prob = float(1.0 - kept_prob)
        kept_dist = AlphabetDistribution.from_probs(kept_symbols, kept_p, normalize=True)
        return DeltaSubsetResult(
            kept=kept_dist,
            dropped_symbols=dropped_symbols,
            kept_probability=kept_prob,
            dropped_probability=dropped_prob,
            threshold=threshold,
        )



