from keras.models import Model
from cpmp_ml.optimizer import OptimizerStrategy
import numpy as np

def validate_model(model: Model, S: int, H: int, N: int, size_states: int, benchmark_csv: str) -> dict:
    pass

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