# Librerias cpmp_ml
from cpmp_ml.utils.functions import generate_random_layout
from cpmp_ml.utils.adapters import DataAdapter
from cpmp_ml.optimizer import OptimizerStrategy
from cpmp_ml.utils.generator import random_perturbate_layout
from cpmp_ml.utils.generator import generate_y 
from cpmp_ml.utils.generator import permutate_y

# Librerias externas
from copy import deepcopy
import random
import numpy as np

# Generación de datos con los optimizadores greedy
def generate_data_v1(S: int = -1, 
                     H: int = -1, 
                     N: int = -1, 
                     sample_size: int = 0, 
                     verbose: bool = False, 
                     from_feasible: bool = False, 
                     perms_by_layout: int = -1,
                     moves: int = 5,
                     solver: OptimizerStrategy = None,
                     adapter: DataAdapter = None, **kwargs) -> tuple:
    
    if from_feasible and perms_by_layout == -1:
        raise ValueError("from_feasible is true, but perms_by_layout is invalid")
    
    if S == -1 or H == -1 or N == -1:
        raise ValueError("One of the parameters 'S', 'H' or 'N' is invalid")
    
    if sample_size == 0:
        raise ValueError("The 'sample_size' parameter is 0")
    
    if solver is None:
        raise ValueError("'solver' is None")
    
    if adapter is None:
        raise ValueError("'adapter' is None")
    
    x = []
    y = []
    n = 0

    while n < sample_size:
        if len(x) == sample_size: break

        # Generar el layout random
        lay = generate_random_layout(S, H, N, feasible=from_feasible)

        # Generar perturbación de ser necesario
        if from_feasible: random_perturbate_layout(lay, moves=moves)

        # Analizar el coste
        copy_lay = deepcopy(lay)
        p_cost = solver.solve(np.array([copy_lay]), **kwargs)[0]
        y_ = generate_y(layout=copy_lay, p_cost=p_cost, optimizer= solver, **kwargs)

        if y_ is None: continue

        for k in range(perms_by_layout):
            enum_stacks = list(range(S))
            perm = random.sample(enum_stacks, S)
            copy_lay.permutate(perm)
            y_ = permutate_y(y_, S, perm)

            x.append(adapter.get_ann_state(copy_lay))
            y.append(deepcopy(y_))

            if len(x) == sample_size: break

            n = n + 1
            if n % 5000 == 0: print(n)
            if n >= sample_size: break

    return np.array(x), np.array(y)