from abc import ABC, abstractmethod
import numpy as np

class StochasticSystem(ABC):
    @abstractmethod
    def sample(self, T: int):
        pass

    @abstractmethod
    def log_prob(self, data):
        pass


class Channel(ABC):
    @abstractmethod
    def transmit(self, x: np.ndarray) -> np.ndarray:
        """Transmit input x through the channel, return noisy output"""
        pass