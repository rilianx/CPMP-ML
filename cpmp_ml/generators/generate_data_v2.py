from cpmp_ml.optimizer import OptimizerStrategy
from cpmp_ml.utils.generator import generate_y
from cpmp_ml.utils.adapters import DataAdapter
from cpmp_ml.utils import generate_random_layout
from cpmp_ml.utils import Layout
from copy import deepcopy
import numpy as np
import random

def generate_steps_state(lay: Layout, N: int, 
                         optimizer: OptimizerStrategy, adapter: DataAdapter,
                         max_steps: int) -> tuple:
    cont = 0
    temp_lay = deepcopy(lay)

    p_cost, moves = optimizer.solve(np.array([temp_lay]), max_steps= max_steps)
    if p_cost[0] == -1 and moves[0] is None: return None, None

    lays, labels = [], []
    while lay.unsorted_stacks != 0:
        temp_lay = deepcopy(lay)
        y_ = generate_y(temp_lay, p_cost[0], optimizer, max_steps= max_steps)

        if y_ is None: return None, None
        labels.append(y_)
        lays.append(adapter.get_ann_state(temp_lay))

        lay.move(moves[0][cont])
        cont += 1
        p_cost[0] -= 1

    return lays, labels

# Generación de datos con los optimizadores greedy enviando los pasos intermedios
def generate_data_v2(min_S: int, max_S: int, H: int, 
                     size: int, lb: float, ub: float, 
                     space_between: float, optimizer: OptimizerStrategy,
                     adapter: DataAdapter, verbose: bool = True
                     ) -> dict:
    x, y = np.empty(shape=(size,), dtype= object), np.empty(shape=(size,), dtype= object)
    space = 0
    cont = 0

    while True:
        S = random.randint(min_S, max_S)
        N = S * (H - 2)

        lay = generate_random_layout(S, H, N)
        lays, labels = generate_steps_state(lay, N, optimizer, adapter, max_steps= N * 2)

        if lays is None and labels is None: continue

        lb_size = int(len(lays) * lb)
        ub_size = int(len(lays) * ub)

        data = []
        space = lb_size
        while space < ub_size:
            data.append((lays[int(space)], labels[int(space)]))
            space += space_between

        for layout, label in data:
            if cont == size: break
            if verbose and cont % 100 == 0: print(f'sample_size: {cont}')

            x[cont], y[cont] = layout, label
            cont += 1
        
        if cont == size: break

    return x, y

