import random


def monte_carlo(N : int, E):
    cnt = 0
    for n in range(N):
        E2 = E.copy()
        for e in E:
            u = random.uniform(1,10)
            if u > p