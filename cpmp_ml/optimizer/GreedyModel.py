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

    def solve(self, lays: np.ndarray[Layout], **kwargs) -> tuple:
        costs = -np.ones(lays.shape[0])
        max_steps = kwargs["max_steps"]
        lays_moves = [[] for _ in lays.shape[0]]

        for steps in range(max_steps):
            x = self.__get_valid_data(steps, costs, lays)
            if x.shape[0] == 0: break
            actions = self.__model.predict(x, verbose=False)
            self.__update_cost(actions, costs, lays)

        self.__verify_solutions__(lays, lays_moves)

        return costs, lays_moves
    
    def __verify_solutions__(self, lays:np.ndarray[Layout], lays_moves: list[list]):
        for i in range(lays.shape[0]):
            if lays[i].unsorted_stacks != 0: lays_moves[i] = None

    def __get_valid_data(self, steps:int, costs:np.ndarray, lays:np.ndarray[Layout]) -> np.ndarray:
        x = []
        for i in range(lays.shape[0]):
            if lays[i].unsorted_stacks == 0 and costs[i] == -1: 
                costs[i] = steps

            ann_state = self.__data_adapter.get_ann_state(lays[i])
            x.append(ann_state)

        return np.stack(x)
    
    def __update_cost(self, actions, costs:np.ndarray, lays:np.ndarray[Layout], lays_move: list[list]) -> None:
        k = 0
        for i in range(lays.shape[0]):
            if costs[i] != -1: continue
            act = np.argmax(actions[k])
            move = self.__data_adapter.get_move(act, len(lays[i].stacks))
            lays[i].move(move)
            lays_move[i].append(move)
            k+=1