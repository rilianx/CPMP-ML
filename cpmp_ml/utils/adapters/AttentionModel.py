from cpmp_ml.utils.adapters import DataAdapter
from cpmp_ml.utils import Layout
import numpy as np

class AttentionModel(DataAdapter):
    def __init__(self, S:int) -> None:
        self.__S = S

    def get_ann_state(self, lay: Layout):
        S=len(lay.stacks) 
        
        b = 2. * np.ones([S,lay.H + 1]) 
        for i,j in enumerate(lay.stacks):
            b[i][lay.H-len(j) + 1:] = [k/lay.total_elements for k in j]
            b[i][0] = lay.is_sorted_stack(i)
        b.shape=(S,(lay.H + 1))
        return b

    def get_move(self, act: int) -> tuple:
        k=0
        for i in range(self.__S):
            for j in range(self.__S):
                if(i==j): continue
                if k==act: return (i,j)
                k+=1
        return (None, None)

    def get_layout_from_ann_state(self, ann_state: np.ndarray[np.ndarray], S: int, H: int, N: int) -> Layout:
        elements = 0
        for i in range(ann_state.shape[0]):
            for k in range(1, ann_state.shape[1]):
                if ann_state[i][k] != 2:
                    elements+=1

        stacks = []
        for i in range(ann_state.shape[0]):
            stack = [int(ann_state[i][k] * elements) for k in range(1, ann_state.shape[1]) if ann_state[i][k] != 2]
            stacks.append(stack)

        return Layout(stacks=stacks, H=H)
