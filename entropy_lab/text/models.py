from collections import Counter
from typing import List

class ReferenceLanguageModel:
    """
    Simple reference model: p(word) estimated from a reference corpus (human text).

    We use Laplace smoothing:
        p(w) = (count(w) + alpha) / (N + alpha*(V+1))
    
    where:
        N = total tokens in reference corpus
        V = vocabulary size in reference corpus
        +1 = reserve one "unknown bucket"    
    """

    def __init__(self, reference_tokens: List[str], alpha: float = 1.0):
        if len(reference_tokens) == 0:
            raise ValueError("Reference token list is empty.")
        self.alpha = float(alpha)
        self.counts = Counter(reference_tokens)
        self.N = sum(self.counts.values())
        self.V = len(self.counts)

    def p(self, token: str) -> float:
        count_w = self.counts.get(token, 0)
        numerator = count_w + self.alpha
        denominator = self.N + self.alpha * (self.V + 1)
        return numerator / denominator
