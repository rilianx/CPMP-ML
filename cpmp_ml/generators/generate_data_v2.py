from cpmp_ml.optimizer import OptimizerStrategy
from cpmp_ml.utils.generator import generate_y
from cpmp_ml.utils.adapters import DataAdapter
from cpmp_ml.utils import generate_random_layout
from cpmp_ml.utils import Layout
from multiprocessing import Pool
from copy import deepcopy
import numpy as np
import random

def generate_steps_state(lay: Layout,
                         optimizer: OptimizerStrategy, adapter: DataAdapter,
                         max_steps: int, lb: float) -> tuple:
    cont = 0
    temp_lay = deepcopy(lay)

    p_cost, moves = optimizer.solve(np.array([temp_lay]), max_steps= max_steps)
    if p_cost[0] == -1: return None, None

    lays, labels = [], []
    while lay.unsorted_stacks != 0:
        temp_lay = deepcopy(lay)
        y_ = generate_y(temp_lay, p_cost[0], optimizer, max_steps= max_steps)
        if y_ is None: return None, None

        labels.append(y_)
        lays.append(adapter.get_ann_state(lay))

        lay.move(moves[0][cont])
        cont += 1
        p_cost[0] -= 1

    lb_size = int(len(lays) * lb)

    return lays[lb_size:], labels[lb_size:]

def process_data(x):
    return generate_steps_state(x[0], x[1], x[2], x[3], x[4])

def generate_data_v2(min_S: int, max_S: int, 
                     H: int, size: int, lb: int, 
                     optimizer: OptimizerStrategy, 
                     adapter: DataAdapter, 
                     batch_size: int = 32,
                     verbose: bool = True) -> tuple:
    x, y = [], []

    while True:
        r_stacks = [random.randint(min_S, max_S) for _ in range(batch_size)]
        batch = [(generate_random_layout(r_stacks[i], H, r_stacks[i] * (H - 2)), optimizer, 
                  adapter, (r_stacks[i] * (H - 2)) * 2, lb) for i in range(batch_size)]

        with Pool() as pool:
            result = pool.map(process_data, batch)

        for i in range(len(result)):
            if result[i][0] is None and result[i][1] is None: continue

            for j in range(len(result[i][0])):
                if len(x) == size: return x, y
                if len(x) % 100 == 0 and verbose: print(len(x))
        
                x.append(result[i][0][j])
                y.append(result[i][1][j])


