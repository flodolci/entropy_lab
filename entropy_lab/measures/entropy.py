import numpy as np
import numpy.typing as npt
import math

from entropy_lab.measures.shannon import shannon_information

def compute_entropy(p: npt.NDArray[np.floating], base: float = 2.0, normalize: bool = False) -> float:
    """ Compute the Shannon entropy of a discrete probability distribution.

    Entropy is calculated as H(p) = -∑ p(x) · log_b(p(x)), where the sum is
    taken over all x with p(x) > 0. Zero-probability events are excluded from
    the sum, consistent with the convention 0 · log(0) = 0.

    Parameters
    ----------
    p : array_like of float
        A 1-D array representing a probability distribution. All values must
        be finite and non-negative. If ``normalize=False``, values must sum
        to 1 within a tolerance of 1e-10; if ``normalize=True``, they are
        rescaled to sum to 1 before computation.
    base : float, optional
        Logarithm base used for the entropy calculation. Common choices are
        2 (bits, default), math.e (nats), and 10 (hartleys/bans).
    normalize : bool, optional
        If True, rescale ``p`` by its sum before computing entropy, allowing
        unnormalized count arrays or histograms to be passed directly.
        Defaults to False.

    Returns
    -------
    float
        Shannon entropy of the distribution in units determined by ``base``.
        Returns 0.0 if all probabilities are zero (degenerate distribution).

    Raises
    ------
    ValueError
        If any value in ``p`` is non-finite (NaN or ±inf).
        If any value in ``p`` is negative.
        If ``normalize=False`` and ``p`` does not sum to 1 within tolerance.
        If ``normalize=True`` and ``p`` sums to 0. """
     
    p = np.asarray(p, dtype=float)
    if np.any(~np.isfinite(p)):
        raise ValueError("Probabilities must be finite.")
    if np.any(p < 0):
        raise ValueError("Probabilities must be >= 0.")
    s = p.sum()
    if normalize:
        if s == 0:
            raise ValueError("Sum of probabilities is 0; can't normalize.")
        p = p / s
        s = 1.0
    if not np.isclose(s, 1.0, rtol=1e-10, atol=1e-10):
        raise ValueError(f"Probabilities must sum to 1 (got {s}).")
    p_pos = p[p > 0]
    if p_pos.size == 0:
        return 0.0
    h = 0.0
    for x in p_pos:
        h += x * shannon_information(x, base)
    return h
