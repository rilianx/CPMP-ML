from cpmp_ml.optimizer import OptimizerStrategy
from cpmp_ml.utils.generator import generate_y
from cpmp_ml.utils.generator import load_simbol
from cpmp_ml.utils.adapters import DataAdapter
from cpmp_ml.utils import generate_random_layout
from cpmp_ml.utils import Layout
from multiprocessing import Pool
from copy import deepcopy
import numpy as np
import random

def generate_steps_state(lay: Layout,
                         optimizer: OptimizerStrategy, 
                         adapter: DataAdapter,
                         max_steps: int, 
                         lb: float) -> tuple:
    try:
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
    except Exception as e:
        print(f"Error al generar datos!")
        return None, None
    except KeyboardInterrupt:
        print(f"Generación de datos interrumpida!")
        return None, None


    return lays[lb_size:], labels[lb_size:]

def process_data(x):
    return generate_steps_state(x[0], x[1], x[2], x[3], x[4])

def generate_data_v2(min_S: int, 
                     max_S: int, 
                     H: int, 
                     size: int, 
                     lb: int, 
                     optimizer: OptimizerStrategy, 
                     adapter: DataAdapter, 
                     batch_size: int = 32,
                     verbose: bool = True,
                     num_threads = 1) -> tuple:
    x, y = [], []

    try: 
        while True:
            r_stacks = [random.randint(min_S, max_S) for _ in range(batch_size)]
            batch = [(generate_random_layout(r_stacks[i], H, r_stacks[i] * (H - 2)), optimizer, 
                    adapter, (r_stacks[i] * (H - 2)) * 2, lb) for i in range(batch_size)]
           
            with Pool(processes= num_threads) as pool:
                result = pool.map(process_data, batch)  

            for i in range(len(result)):
                if result[i][0] is None and result[i][1] is None: continue

                for j in range(len(result[i][0])):
                    if verbose: load_simbol(len(x), size, text="Datos generados: ")

                    if len(x) == size: return x, y
            
                    x.append(result[i][0][j])
                    y.append(result[i][1][j])
    except Exception as e:
        print(f"\nError al generar datos!")
        return np.array(x), np.array(y)
    except KeyboardInterrupt:
        print(f"\nGeneración de datos interrumpida!")
        if len(x) != 0: 
            print(f'Enviando los datos generados hasta el momento...')
            return np.array(x), np.array(y)
        else: return None, None


