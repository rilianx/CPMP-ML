from cpmp_ml.utils import Layout
import random

def read_benchmark_file(route: str, H: int) -> Layout:
    pass

def reachable_height(layout: Layout, i: int) -> int:
    pass

def generate_random_layout(S: int, H: int, N: int, feasible: bool = False) -> Layout:
    stacks = []
    for _ in range(S):
        stacks.append([])
    
    for j in range(N):
        s = random.randint(0, S - 1)
        while len(stacks[s])==H: s=s=random.randint(0, S - 1)
        container_priority = random.randint(1, N)
        if feasible: container_priority = N - j
        stacks[s].append(container_priority)

    return Layout(stacks=stacks, H=H)

