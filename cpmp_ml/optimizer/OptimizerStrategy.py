from abc import abstractmethod
from abc import ABC
from cpmp_ml.utils import Layout
from numpy import ndarray

class OptimizerStrategy(ABC):
    
    @abstractmethod
    def solve(self, lays: ndarray[Layout], **kwargs) -> ndarray:
        pass
