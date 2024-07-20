from cpmp_ml.utils import Layout
from cpmp_ml.optimizer import OptimizerStrategy
from copy import deepcopy
import numpy as np
import random

def costs_to_y(costs: np.ndarray, parent_cost: int) -> None | np.ndarray:
    mincost = np.inf
    y = np.zeros(costs.shape[0])

    filtered_costs = [c for c in costs if c != -1]
    mincost = min(filtered_costs) if filtered_costs else None

    if mincost is None or mincost >= parent_cost: return None

    pos = 0
    for c in costs:
        if c == mincost:
            y[pos] = 1
        
        pos += 1

    return y

def generate_y(layout: Layout, p_cost: int, optimizer: OptimizerStrategy, **kwargs) -> None | np.ndarray:
    S = len(layout.stacks)
    label_size = S * (S - 1)
    temp_lay = deepcopy(layout)
    costs = np.zeros(shape=(label_size, ))
    pos = 0

    for i in range(S):
        for j in range(S):
            if i == j: continue

            temp_lay.move((i, j))
            costs[pos] = optimizer.solve(np.array([temp_lay]), **kwargs)[0]
            pos += 1

    return costs_to_y(costs, p_cost)

def gen_movement_matrix(y: np.ndarray, S: int) -> np.ndarray:
    m = np.zeros(shape = (S, S))
    n = 0

    for i in range(S):
        for j in range(S):
            if i == j: continue

            m[i, j] = y[n]
            n+=1

    return m

def permutate_y(y: np.ndarray, S: int, perm: list):
    m = gen_movement_matrix(y, S)
    m = m[perm].T[perm].T
    A = np.zeros(shape= (S * (S - 1)))
    n = 0

    for i in range(S):
        for j in range(S):
            if i == j: continue

            A[n] = m[i, j]
            n+=1

    return A

def random_perturbate_layout(lay:Layout, moves:int = 5) -> None | ValueError:
    if lay is None:
        raise ValueError("Lay is None")

    S=len(lay.stacks)

    last_moves = []
    for _ in range(moves):
        i = random.randint(0, S - 1)
        j = random.randint(0, S - 1)

        while (i, j) in last_moves or lay.move((i, j)) == None: 
            i = random.randint(0, S - 1)
            j = random.randint(0, S - 1)

        last_moves.append((i, j))