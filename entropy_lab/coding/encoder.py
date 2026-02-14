import numpy as np

from entropy_lab.coding.code import Code

class Encoder:
    def __init__(self):
        pass

    def repetition(self, code: Code, n_repetition: int) -> Code:
        repeated_code_array = np.repeat(code.code_array, n_repetition)
        repeated_code = Code(repeated_code_array)
        return repeated_code