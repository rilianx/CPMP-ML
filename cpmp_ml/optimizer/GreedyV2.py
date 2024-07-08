from cpmp_ml.optimizer import OptimizerStrategy
from cpmp_ml.utils import Layout
import numpy as np

class GreedyV2(OptimizerStrategy):
    def __init__(self, params:list = [2.0, 2.0, 4, 2.1, 2]):
        self.__params = params

    def solve(self, lays: np.ndarray[Layout], **kwargs):
        self.__max_steps = kwargs["max_steps"]

        costs = -np.ones(lays.shape[0])
        for k in range(lays.shape[0]):
            steps = self.__greedy(lays[k])
            costs[k]=steps
        return costs

    def __greedy(self, lay:Layout) -> int:
        steps = 0
        while lay.unsorted_stacks > 0 and steps < self.__max_steps:
            actions = lay.get_actions()

            best_ev = float("-inf"); best_action=None
            for action in actions:
                ev = self.__eval_action(lay, action, self.__params)
                if ev > best_ev:
                    best_ev=ev
                    best_action=action

            if best_action is not None:
                lay.move(best_action)
            else:
                return -1
            steps +=1

        if lay.unsorted_stacks==0:
            return steps
        return -1
    
    def __eval_action(self, lay:Layout, action:tuple, params:list):
        s_o, s_d = action
        g_s_d = lay.gvalue(s_d)
        g_s_o = lay.gvalue(s_o)
        c = lay.stacks[s_o][-1]

        if lay.is_BG_action(action):
            diff = g_s_d - g_s_o
            if lay.reduced_stack == -1:
                return 100 - diff

        if lay.reduced_stack == s_o or lay.reduced_stack == -1:
            top_d = lay.gvalue(s_d)

            if lay.is_sorted_stack(s_d) and c <= top_d:  # xg
                eval_dest_stack = -top_d  # minimum difference between c and top_d is preferred
            elif not lay.is_sorted_stack(s_d) and c >= top_d:  # xb
                eval_dest_stack = -10**params[0] + top_d  # minimum difference between c and top_d is preferred
            elif lay.is_sorted_stack(s_d):  # xb
                eval_dest_stack = -10**params[1] - len(lay.stacks[s_d])  # - top_d
            else:
                eval_dest_stack = -10**params[2] - 10**params[3]*len(lay.stacks[s_d]) - top_d

            # Factor in remaining containers in the destination stack
            if len(lay.stacks[s_d]) > 1:
                next_container = lay.stacks[s_d][-2]
                if next_container > c:
                    eval_dest_stack -= 10**params[4]  # Penalize this action


            stack_len_multiplier = 1 + len(lay.stacks[s_o]) / lay.H  # Factor in stack length dynamically
            return stack_len_multiplier * eval_dest_stack

        return float("-inf")