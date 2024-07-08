from cpmp_ml.utils.adapters import DataAdapter
from cpmp_ml.utils import Layout
import numpy as np

# Deja último contenedor en el top del arreglo

class LinealModel(DataAdapter):
    def __init__(self, S:int, H:int):
        self.__S = S
        self.__H = H

    def get_ann_state(self, lay: Layout) -> np.ndarray:
        S = len(lay.stacks)
        b = 2. * np.ones([S, lay.H+1])
        for i,j in enumerate(lay.stacks):
            b[i][lay.H-len(j)+1:] = [k/lay.total_elements for k in j]
            b[i][0] = lay.is_sorted_stack(i)

        mtype = []
        for i in range(5):
            for j in range(5):
                if i==j: continue
                m = lay.move((i,j))

                if m!=None:
                    if lay.is_sorted_stack(j): mtype.append(1.)
                    else: mtype.append(0.)
                    m = lay.move((j,i)); lay.steps-=2
                    if lay.is_sorted_stack(i): mtype.append(1.)
                    else: mtype.append(0.)
                else:
                    mtype.append(-1.); mtype.append(-1.)
        
        b.shape = (S*(lay.H+1),)
        b = np.concatenate((b,np.array(mtype)))

        return b
    
    def get_move(self, act:int) -> tuple:
        k=0
        for i in range(self.__S):
            for j in range(self.__H):
                if i==j: continue
                if k==act: return (i,j)
                k+=1
    
    def get_layout_from_ann_state(self, ann_state: np.ndarray[np.ndarray], S: int, H: int, N: int) -> Layout:
        return 
    