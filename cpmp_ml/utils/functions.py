from cpmp_ml.utils import Layout
import random
import sys

def read_benchmark_file(route: str, H: int) -> Layout:
    with open(route) as file:
        S, C = [int(x) for x in next(file).split()]

        stacks = []
        for line in file:
            stack = [int(x) for x in line.split()[1::]]
            stacks.append(stack)
        
        lay = Layout(stacks, H)

    return lay

def reachable_height(layout: Layout, i: int) -> int:
    if not layout.is_sorted_stack(i): return -1

    top = layout.gvalue(i)
    S = len(layout.stacks)
    h = len(layout.stacks[i])
    all_stacks = True

    if h == layout.H: return h

    for k in range(S):
        if k == i: continue
        if layout.is_sorted_stack(k): continue

        stack_k = layout.stacks[k]
        unsorted = len(stack_k) - layout.sorted_elements[k]
        prev = 1000

        for j in range(1, unsorted + 1):
            if stack_k[-j] <= prev and stack_k[-j] <= top:
                h += 1
                if h == layout.H: return h

                prev = stack_k[-j]
            elif j == 1: 
                all_stacks = False
                break
    
    if all_stacks: return layout.H
    return h

def generate_random_layout(S: int, H: int, N: int, feasible: bool = False) -> Layout:
    stacks = [[] for _ in range(S)]
    
    for j in range(N):
        s = random.randint(0, S - 1)

        while len(stacks[s]) == H: 
            s = random.randint(0, S - 1)

        g = random.randint(1, N)
        if feasible: g = N - j

        stacks[s].append(g)

    return Layout(stacks, H)

