from cpmp_ml.optimizer import OptimizerStrategy
from cpmp_ml.utils.generator import generate_y
from cpmp_ml.utils.adapters import DataAdapter
from cpmp_ml.utils import generate_random_layout
from cpmp_ml.utils import Layout
from copy import deepcopy
import numpy as np
import random

def generate_steps_state(lay: Layout,
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

        if y_ is None and len(lays) == 0: return None, None

        labels.append(y_)
        lays.append(adapter.get_ann_state(lay))

        lay.move(moves[0][cont])
        cont += 1
        p_cost[0] -= 1

    return lays, labels

# Generación de datos con los optimizadores greedy enviando los pasos intermedios
def generate_data_v2(min_S: int, max_S: int, H: int, 
                     size: int, lb: float, 
                     optimizer: OptimizerStrategy,
                     adapter: DataAdapter, verbose: bool = True
                     ) -> dict:
    x, y = [], []

    while True:
        S = random.randint(min_S, max_S)
        N = S * (H - 2)

        lay = generate_random_layout(S, H, N)
        lays, labels = generate_steps_state(lay, optimizer, adapter, max_steps= N * 2)

        if lays is None or labels is None: continue

        lb_size = int(len(lays) * lb)

        data = zip(lays[lb_size:], labels[lb_size:])
        for state, label in data:
            if len(x) == size: return x, y
            if verbose and len(x) % 100 == 0: print(len(x))
            
            x.append(state)
            y.append(label)

    return x, y

