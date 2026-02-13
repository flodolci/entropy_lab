from abc import ABC, abstractmethod

class StochasticSystem(ABC):
    @abstractmethod
    def sample(self, T: int):
        pass

    @abstractmethod
    def log_prob(self, data):
        pass