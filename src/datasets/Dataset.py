from abc import ABC, abstractmethod

class Dataset(ABC):
    def __init__(self):
        super().__init__()
        
    @abstractmethod
    def _get_dataset(self):
        pass