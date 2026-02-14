import numpy as np

class Code:
    def __init__(self, code_array: list):
        self.code_array = np.array(code_array, dtype=int)
    
    def repetition(self, n_repetition):
        return np.repeat(self.code_array, n_repetition)