from cpmp_ml.optimizer import OptimizerStrategy
from cpmp_ml.utils.generator import generate_y
from cpmp_ml.utils.adapters import DataAdapter
from cpmp_ml.utils import generate_random_layout
from cpmp_ml.utils import Layout
from copy import deepcopy
import numpy as np
import random

def generate_steps_state(lay: Layout, N: int, 
                         optimizer: OptimizerStrategy, adapter: DataAdapter
                         ) -> tuple:
    cont = 0
    temp_lay = deepcopy(lay)

    p_cost, moves = optimizer.solve(np.array([temp_lay]), N * 2)
    if p_cost == -1 and moves is None: return None, None

    lays, labels = np.empty(shape=(p_cost, )), np.empty(shape=(p_cost, ))
    while lay.unsorted_stacks != 0:
        temp_lay = deepcopy(lay)
        y_ = generate_y(temp_lay, p_cost, max_steps = N * 2)

        if y_ is None: return None, None
        labels[cont] = y_
        lays[cont] = adapter.get_ann_state(temp_lay)

        lay.move(moves[cont])
        cont += 1
        p_cost -= 1

    return lays, labels

# Generación de datos con los optimizadores greedy enviando los pasos intermedios
def generate_data_v2(min_S: int, max_S: int, H: int, 
                     size: int, lb: float, ub: float, 
                     space_between: float, optimizer: OptimizerStrategy,
                     adapter: DataAdapter, verbose: bool = True
                     ) -> dict:
    x, y = np.empty(shape=(size,)), np.empty(shape=(size,))
    space = 0
    cont = 0

    while True:
        S = random.randint(min_S, max_S)
        N = S * (H - 2)

        lay = generate_random_layout(S, H, N)
        lays, labels = generate_steps_state(lay, N, optimizer, adapter)

        if lays is None and labels is None: continue

        lb_size = int(lays.shape[0] * lb)
        ub_size = int(lays.shape[0] * ub)
        space += space_between 

        data = zip(lays[lb_size: ub_size: int(space)], labels[lb_size: ub_size: int(space)])

        for layout, label in data:
            if cont == size: break
            if verbose: print(f'sample_size: {cont}')

            x[cont], y[cont] = layout, label
            cont += 1
        
        if cont == size: break

    return x, y

