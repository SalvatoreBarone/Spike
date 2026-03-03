from abc import ABC, abstractmethod

class GenericSNN(ABC):
    def __init__(self):
        self.model = None
        self.q_aware_model = None

    @abstractmethod
    def _get_model(self):
        pass
    