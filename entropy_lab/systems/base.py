from abc import ABC, abstractmethod
import numpy as np

from entropy_lab.coding.code import Code

class StochasticSystem(ABC):
    @abstractmethod
    def sample(self, T: int):
        pass

    @abstractmethod
    def log_prob(self, data):
        pass


class Channel(ABC):
    @abstractmethod
    def transmit(self, code: Code, noise_level) -> Code:
        """Transmit input code through the channel with noise level noise_level, return noisy output"""
        pass