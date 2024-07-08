from cpmp_ml.optimizer import OptimizerStrategy
from cpmp_ml.utils.adapters import DataAdapter
from cpmp_ml.utils import generate_random_layout
from cpmp_ml.generators.functions import permutate_y

from copy import deepcopy
import numpy as np
import random

#Generación de datos con greedy + model
def generate_data_v3(
        solver: OptimizerStrategy = None,
        adapter: DataAdapter = None,
        S: int = -1,
        H: int = -1,
        N: int = -1,
        sample_size: int = 0,
        batch_size: int = 0,
        perms_by_layout: int = 20,
    ) -> tuple:

    if adapter is None: 
        raise ValueError("Adapter is None")
    if solver is None: raise ValueError("Solver is None")
    if sample_size == 0: raise ValueError("The parameter 'sample_size' is 0")
    if batch_size == 0 or batch_size > sample_size: raise ValueError("batch_size is invalid")
    if S == -1 or H == -1 or N == -1: 
        raise ValueError("One of the parameters 'S', 'H' or 'N' is invalid")
    
    x = []
    y = []
    n = 0

    while True:
        lays = []
        for i in range(batch_size):
            lays.append(generate_random_layout(S=S, H=H, N=N))

        lays_copy = deepcopy(lays)
        costs = solver.solve(np.array(lays))

        # for each lay we generate children clays
        child_lays = []
        for p in range(batch_size):
            for i in range(S):
                for j in range(S):
                    if i == j: continue
                    child_lay = deepcopy(lays_copy[p])
                    child_lay.move((i, j))
                    child_lays.append(child_lay)

        child_costs = solver.solve(np.array(child_lays))

        # for each parent to verify the existence of solutions
        for p in range(batch_size):
            mincost = np.inf

            # Get min cost from childs
            for c in range(p * (S * (S - 1)), (p + 1) * (S * (S - 1))):
                if child_costs[c] != -1 and child_costs[c] < mincost:
                    mincost = child_costs[c]

            if costs[p] != -1 and mincost >= costs[p]: continue

            A = []
            for c in range(p * (S * (S - 1)), (p + 1) * (S * (S - 1))):
                if child_costs[c] != -1 and child_costs[c] == mincost:
                    A.append(1)
                else:
                    A.append(0)

            if sum(A) > 0:  # otherwise no action was succesful, we simply discard the data
                for _ in range(perms_by_layout):
                    enum_stacks = list(range(S))
                    perm = random.sample(enum_stacks, S)
                    lays_copy[p].permutate(perm)
                    A = permutate_y(A, S, perm)

                    x.append(adapter.get_ann_state(lays_copy[p]))
                    y.append(deepcopy(A))
                    if len(x) == sample_size: return np.array(x), np.array(y)
