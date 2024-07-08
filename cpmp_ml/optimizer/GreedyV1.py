from cpmp_ml.optimizer import OptimizerStrategy
from cpmp_ml.utils import Layout
import numpy as np

class GreedyV1(OptimizerStrategy):
    def __init__(self):
        pass

    def solve(self, layouts: np.ndarray[Layout], **kwargs):
        costs = -np.ones(len(layouts))
        for k in range(len(layouts)):
            steps = self.__greedy(layouts[k])
            costs[k]=steps
        return costs
    
    def __greedy(self, layout: Layout):
        steps = 0
        while layout.unsorted_stacks>0:
            bg_move=self.__select_bg_move(layout)
            if bg_move is not None:
                layout.move(bg_move)
            else:
                return -1 # no lo resuelve
            steps +=1

        if layout.unsorted_stacks==0: 
            return steps
        return -1
    
    def __select_bg_move(self, layout:Layout):
        bg_move = None
        S=len(layout.stacks)
        min_diff = 100
        for s_o in range(S):
            for s_d in range(S):
                if self.__is_valid_BG_move(layout, s_o, s_d):
                    diff = layout.gvalue(s_d) - layout.gvalue(s_o)
                    if min_diff > diff:
                        min_diff = diff
                        bg_move = (s_o,s_d)
        return bg_move
    
    def __is_valid_BG_move(self, layout: Layout, s_o:int, s_d:int):
        if (s_o != s_d  and len(layout.stacks[s_o]) > 0
            and  len(layout.stacks[s_d]) < layout.H
            and layout.is_sorted_stack(s_o)==False
            and layout.is_sorted_stack(s_d)==True
            and layout.gvalue(s_o) <= layout.gvalue(s_d)):
            return True

        return False