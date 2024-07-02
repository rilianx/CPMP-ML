from cpmp_ml.utils import Layout
import numpy as np
import random

def costs_to_y(costs: np.ndarray, parent_cost: np.ndarray) -> np.ndarray:
    pass

def generate_y(layout: Layout, p_cost: int) -> np.ndarray:
    pass

def gen_movement_matrix(y: np.ndarray, S: int) -> np.ndarray:
    m = np.zeros(shape = (S, S))
    n=0

    for i in range(S):
        for j in range(S):
            if i == j: 
                continue

            m[i, j] = y[n]
            n+=1

    return m

def permutate_y(y: np.ndarray, S: int, perm: int):
    m = gen_movement_matrix(y, S)
    # print(m)
    m = m[perm].T[perm].T
    # print(m)
    A=np.zeros(shape= (S*(S-1)))
    n=0

    for i in range(S):
        for j in range(S):
            if i == j: continue

            A[n] = m[i, j]
            n+=1

    return A

def random_perturbate_layout(lay: Layout, moves: int = 5) -> None:
    S = len(lay.stacks)

    last_moves = []
    for _ in range(moves):
        i = random.randint(0, S-1)
        j = random.randint(0, S-1)

        while (i, j) in last_moves or lay.move((i, j)) == None: 
            i = random.randint(0, S-1)
            j = random.randint(0, S-1)

        last_moves.append((i, j))