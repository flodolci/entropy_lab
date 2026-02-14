import numpy as np

from entropy_lab.coding.code import Code

class Decoder:
    def __init__(self):
        pass
    
    def majority_vote(self, noisy_code: Code, n_repetition: int) -> Code:
        blocks = noisy_code.code_array.reshape(-1, n_repetition)
        ones_count = blocks.sum(axis=1)
        decoded = (ones_count > n_repetition / 2).astype(int)
        decoded_code = Code(decoded)
        return decoded_code
