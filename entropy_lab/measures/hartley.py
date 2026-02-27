import math

def raw_bit_content(alphabet_size: int) -> float:
    """
    Raw bit content H0(X) = log2(|A_X|).

    Parameters
    ----------
    alphabet_size : int
        Size of the alphanet |A_X| (must be >= 1).

    Returns
    -------
    float
        H0 in bits.
    """
    if not isinstance(alphabet_size, int) or alphabet_size < 1:
        raise ValueError("alphabet_size must be an integer >= 1.")
    return math.log2(alphabet_size)