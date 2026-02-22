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


def compute_joint_entropy(
        p_xy: npt.NDArray[np.floating], 
        base: float = 2.0, 
        normalize: bool = False
    ) -> float:
    """
    Compute the joint Shannon entropy H(X,Y) of a discrete joint distribution.

    Parameters
    ----------
    p_xy : array_like of float
        A 2-D array representing the joint probability table p(x,y)
    base : float, optional
        Logarithm base used for the entropy calculation (default: 2)
    normalize: bool, optional
        If True, rescale p_xy by its sum before computing entropy.

    Returns
    -------
    float
        Joint entropy H(X,Y) in units determined by `base`.
    """
    p_xy = np.asarray(p_xy, dtype=float)
    if p_xy.ndim != 2:
        raise ValueError(f"`p_xy` must be a 2-D array (got shape {p_xy.shape}).")
    return compute_entropy(p_xy.ravel(), base=base, normalize=normalize)


def compute_kl_divergence(
        p: npt.NDArray[np.floating], 
        q: npt.NDArray[np.floating], 
        base: float = 2.0
    ) -> float:
    """
    Compute the discrete Kullback-Leibler divergence D_KL(P||Q) 
    for two probability distributions p and q.

    Parameters
    ----------
    p : npt.NDArray
        True/reference distribution (1D array of probabilities).
    q : npt.NDArray
        Approximation/model distribution (1D array of probabilities).
    base : gfloat
        Logarithm base (2 -> bits)

    Returns
    -------
    float
        KL divergence D_KL(P||Q)
    """
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    if p.shape != q.shape:
        raise ValueError("p and q must have the same shape")
    if np.any(p < 0) or np.any(q < 0):
        raise ValueError("Probabilities must be non-negative")
    # normalize
    p_sum = p.sum()
    q_sum = q.sum()
    if p_sum == 0 or q_sum == 0:
        raise ValueError("Distribution must have positive total mass")
    p = p / p_sum 
    q = q / q_sum 
    # if q_i == 0 where p_i > 0, divergence is infinite
    if np.any((p > 0) & (q == 0)):
        return np.inf
    # Only compute where p > 0 
    mask = p > 0 
    return np.sum(p[mask] * (np.log(p[mask] / q[mask]) / np.log(base)))

