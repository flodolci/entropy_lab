import math

def shannon_information(p, base = 2.0):
    """
    Shannon information content (surprisal):
        h(x) = log_base(1 / P(x)) = -log_base(P(x))

    p : probability of the outcome
    base : logarithm base (2 -> bits, e -> nats, 10 -> hartleys)
    """
    if p is None:
            raise ValueError("Provide p=...")
    if p < 0.0:
        raise ValueError(f"Probability must be >= 0, got {p}.")
    if p == 0.0:
        raise ValueError("P(x)=0 => information content is infinite (or undefined).")
    if p > 1.0:
        raise ValueError(f"Probability must be <= 1, got {p}.")
    return - math.log(p, base)
