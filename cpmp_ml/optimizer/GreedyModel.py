from cpmp_ml.optimizer import OptimizerStrategy
from cpmp_ml.utils import Layout
from cpmp_ml.utils.adapters import DataAdapter
from keras.models import Model
import numpy as np

class GreedyModel(OptimizerStrategy):

    def __init__(self, model:Model = None, 
                 data_adapter: DataAdapter = None):
        if model is None is None or data_adapter is None:
            return ValueError("Some parameter was not given by argument.")
        self.__model = model
        self.__data_adapter = data_adapter

    def solve(self, lays: np.ndarray[Layout], **kwargs):
        costs = -np.ones(lays.shape[0])
        max_steps = kwargs["max_steps"]
        for steps in range(max_steps):
            x = self.__get_valid_data(steps, costs, lays)
            if x.shape[0] == 0: break
            actions = self.__model.predict(x, verbose=False)
            print(actions)
            self.__update_cost(actions, costs, lays)

        return costs

    def __get_valid_data(self, steps:int, costs:np.ndarray, lays:np.ndarray[Layout]) -> np.ndarray:
        x = []
        for i in range(lays.shape[0]):
            if lays[i].unsorted_stacks == 0 and costs[i] == -1: 
                costs[i] = steps

            ann_state = self.__data_adapter.get_ann_state(lays[i])
            x.append(ann_state)

        return np.stack(x)
    
    def __update_cost(self, actions, costs:np.ndarray, lays:np.ndarray[Layout]) -> None:
        k = 0
        for i in range(lays.shape[0]):
            if costs[i] != -1: continue
            act = np.argmax(actions[k])
            move = self.__data_adapter.get_move(act, len(lays[i].stacks))
            lays[i].move(move)
            k+=1