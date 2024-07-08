from keras.models import Model
from cpmp_ml.optimizer import OptimizerStrategy
from cpmp_ml.optimizer import GreedyModel
from cpmp_ml.utils.adapters import DataAdapter
from cpmp_ml.utils import generate_random_layout
from statistics import mean
from statistics import median
from copy import deepcopy
import numpy as np

def validate_model(model: Model, optimizer: OptimizerStrategy, data_adapter: DataAdapter, S: int, H: int, N: int, size_states: int, benchmark_csv: str = None, **kwargs) -> dict:
    optimizer_model = GreedyModel(model, data_adapter)

    if benchmark_csv is None:
        lays = [generate_random_layout(S, H, N) for _ in range(size_states)]

    lays1 = deepcopy(lays)
    costs1 = optimizer.solve(np.array(lays1), **kwargs)
    costs2 = optimizer_model.solve(np.array(lays), **kwargs)

    valid_costs1 = [v for v in costs1 if v!=-1]
    valid_costs2 = [v for v in costs2 if v!=-1]

    results_model = len(valid_costs1) / size_states * 100.
    results_greedy = len(valid_costs2) / size_states * 100.

    if len(valid_costs1)>0:
        print(f"success ann model (%): {results_model}") 
        print(f"mean steps: {mean(valid_costs1)}")
        print(f"median steps: {median(valid_costs1)}")
        #print(f"stdesv steps: {stdev(valid_costs1)}")
        print(f"min steps: {min(valid_costs1)}")
        print(f"max steps: {max(valid_costs1)}")
        print('')
    if len(valid_costs2)==0:
        print("success heuristic (%):", results_greedy)
    else:
        print("success heuristic (%):", results_greedy, mean(valid_costs2))
        print(f"mean steps: {mean(valid_costs2)}")
        print(f"median steps: {median(valid_costs2)}")
        #print(f"stdesv steps: {stdev(valid_costs2)}")
        print(f"min steps: {min(valid_costs2)}")
        print(f"max steps: {max(valid_costs2)}")
        print('')

    return results_model, results_greedy

def cosine_Similarity(y_predict, y_test):
    """
    This function is to verify if the values
    predicted by a multiclass classification deep learning
    mechanism are correct or not.

    Input:

        y_predict (list): Values predicted by the machine 
                          learning.
        y_test (list): Actual values for each case.
    
    Return:
        float: Proportion of correctly predicted values over 
        the total number of cases.
    """
    size = len(y_predict)
    suma = 0

    for i in range(size):
        result = np.dot(y_predict[i], y_test[i]) / (np.linalg.norm(y_predict[i]) * np.linalg.norm(y_test[i]))
        suma += result
    
    return suma / size