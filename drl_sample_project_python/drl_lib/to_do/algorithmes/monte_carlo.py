import random
import pandas as pd

def monte_carlo_es(N : int, E : pd.DataFrame):
    S = [0, 1, 2]
    A = [0,1,2,3,4,5,6,7,8]
    cnt = 0
    for n in range(N):
        E2 = E.copy()
        for e in E:
            u = random.uniform(1,10)
            if u > p(e)
                E2.drop(e)
        if