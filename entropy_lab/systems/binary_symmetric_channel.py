import numpy as np

from entropy_lab.systems.base import Channel
from entropy_lab.coding.code import Code

class BSC(Channel):
    def __init__(self):
        pass

    def transmit(self, code: Code, noise_level):
        flips = np.random.rand(*code.code_array.shape) < noise_level
        transmitted_code_array = code.code_array ^ flips.astype(code.code_array.dtype)
        transmitted_code = Code(transmitted_code_array)
        return transmitted_code



