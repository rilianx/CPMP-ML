from cpmp_ml.utils.Layout import Layout
from cpmp_ml.optimizer import OptimizerStrategy

from copy import deepcopy
import random
import numpy as np

def random_perturbate_layout(lay:Layout, moves:int = 5) -> None | ValueError:
    if lay is None:
        return ValueError("Lay is None")

    S=len(lay.stacks)

    last_moves = []
    for _ in range(moves):
        i = random.randint(0, S - 1)
        j = random.randint(0, S - 1)

        while (i,j) in last_moves or lay.move((i,j)) == None: 
            i = random.randint(0, S - 1)
            j = random.randint(0, S - 1)

        last_moves.append((i,j))

def generate_y(lay: Layout, 
               p_cost:int, 
               solver: OptimizerStrategy) -> None | np.ndarray:
    S = len(lay.stacks)
    l = deepcopy(lay)
    n = 0
    costs = []
    for i in range(S):
        for j in range(S):
            if(i!=j):
                l.move((i,j))
                costs.append(solver.solve(np.array([l]))[0])
                l = deepcopy(lay)
                n += 1

    return costs_to_y(costs, p_cost)

def costs_to_y(costs:list, parent_cost:int) -> None | np.ndarray:
    mincost = np.inf
    y = []
    for c in costs:
        if c != -1 and c < mincost:
            mincost = c

    if c != -1 and mincost >= parent_cost:
        return None

    for c in costs:
        if c == mincost:
            y.append(1)
        else:
            y.append(0)
    return np.array(y)

def permutate_y(y: np.ndarray, S: int, perm: list) -> np.ndarray:
    m = gen_movement_matrix(y, S)
    m = m[perm].T[perm].T
    A = np.zeros(shape= (S*(S-1)))
    n = 0
    for i in range(S):
        for j in range(S):
            if i == j: continue
            A[n] = m[i, j]
            n += 1
    return A

def gen_movement_matrix(y: np.ndarray, S: int) -> np.ndarray[np.ndarray]:
    m = np.zeros(shape = (S, S))
    n=0
    for i in range(S):
        for j in range(S):
            if i == j: 
                continue
            m[i, j] = y[n]
            n+=1
    return m