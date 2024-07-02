from cpmp_ml.utils.adapters import DataAdapter
from cpmp_ml.utils import Layout
import numpy as np

class AttentionModel(DataAdapter):
    def get_ann_state(lay: Layout):
        pass

    def get_move(act: np.ndarray) -> tuple:
        pass

    def get_layout_from_ann_state(ann_state: np.ndarray[np.ndarray], S: int, H: int, N: int) -> Layout:
        pass