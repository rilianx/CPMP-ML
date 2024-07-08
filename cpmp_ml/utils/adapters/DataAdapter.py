from abc import abstractmethod
from abc import ABC
from cpmp_ml.utils import Layout
from numpy import ndarray

class DataAdapter(ABC):
    @abstractmethod
    def get_ann_state(self, lay: Layout) -> ndarray:
        pass

    @abstractmethod
    def get_move(self, act:int, S: int) -> tuple:
        pass
    
    @abstractmethod
    def get_layout_from_ann_state(self, ann_state: ndarray[ndarray], S: int, H: int, N: int) -> Layout:
        pass