import numpy as np

from entropy_lab.coding.alphabet import AlphabetDistribution

def compute_kl_divergence(
        dist_1: AlphabetDistribution, 
        dist_2: AlphabetDistribution, 
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
    p = dist_1.p
    q = dist_2.p
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


def js_divergence(p: "AlphabetDistribution", q: "AlphabetDistribution", base: float = 2.0) -> float:
    """
    Jensen-Shannon divergence.
    Requires identical alphabet order.
    """
    if p.symbols != q.symbols:
        raise ValueError("p and q must have identical symbol tuples (same alphabet & order).")

    m_probs = 0.5 * (p.p + q.p)
    m = AlphabetDistribution.from_probs(p.symbols, m_probs, normalize=True)

    return 0.5 * p.kl_divergence(m, base=base) + 0.5 * q.kl_divergence(m, base=base)