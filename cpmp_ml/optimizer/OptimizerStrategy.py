from abc import abstractmethod
from abc import ABC
from cpmp_ml.utils import Layout
from numpy import ndarray

class OptimizerStrategy(ABC):
    
    @abstractmethod
    def solve(lays: ndarray[Layout]):
        pass