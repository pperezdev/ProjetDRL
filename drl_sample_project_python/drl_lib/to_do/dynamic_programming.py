import random

from ..do_not_touch.mdp_env_wrapper import Env1
from ..do_not_touch.result_structures import ValueFunction, PolicyAndValueFunction
import numpy as np


class LineWorldEnv:
    def __init__(self):
        self.S = [0, 1, 2, 3, 4, 5, 6]
        self.A = [0, 1]
        self.R = [-1.0, 0.0, 1.0]

        self.p = np.zeros((len(self.S), len(self.A), len(self.S), len(self.R)))

        for s in self.S[1:-1]:
            if s == 1:
                self.p[s, 0, s - 1, 0] = 1.0
            else:
                self.p[s, 0, s - 1, 1] = 1.0

            if s == 5:
                self.p[s, 1, s + 1, 2] = 1.0
            else:
                self.p[s, 1, s + 1, 1] = 1.0


def policy_evaluation(pi: np.ndarray, lwe: LineWorldEnv):
    theta = 0.0000002

    V = np.random.random((len(lwe.S),))
    V[0] = 0.0
    V[6] = 0.0
    # print("type de V" ,type(V),"and :",V)

    while True:
        delta = 0
        for s in lwe.S:
            v = V[s]
            V[s] = 0.0
            for a in lwe.A:
                total = 0.0
                for s_p in lwe.S:
                    for r in range(len(lwe.R)):
                        total += lwe.p[s, a, s_p, r] * (lwe.R[r] + 0.999 * V[s_p])
                total *= pi[s, a]
                V[s] += total
            delta = max(delta, np.abs(v - V[s]))
        if delta < theta:
            break
    return V

def policy_evaluation_on_line_world() -> ValueFunction:
    """
    Creates a Line World of 7 cells (leftmost and rightmost are terminal, with -1 and 1 reward respectively)
    Launches a Policy Evaluation Algorithm in order to find the Value Function of a uniform random policy
    Returns the Value function (V(s)) of this policy
    """

    lwe = LineWorldEnv()

    # random_pi = np.ones((len(lwe.S), len(lwe.A))) / len(lwe.A)
    # random_pi = np.ones( 1 / (len(lwe.S) - len(lwe.A)))
    
    random_pi = np.ones((len(lwe.S), len(lwe.A))) * 1 / (len(lwe.S) - len(lwe.A))
    print(random_pi)

    V = policy_evaluation(random_pi, lwe)

    result = {i: v for i, v in enumerate(V)}

    return result


def action_values(lwe: LineWorldEnv, state, V, theta=0.0000002):
    A = np.zeros(len(lwe.A))
    for i in lwe.A:
        for proba, next_state, reward in lwe.p[state][i]:
            A[i] += proba * (reward + theta * V[next_state])
    return A

def policy_improvement(lwe: LineWorldEnv, V):
    #policy = np.ones((len(lwe.S), len(lwe.A))) * 1 / (len(lwe.S) - len(lwe.A))
    policy_stable = True

    pi_dict = dict()

    for i, s in enumerate(lwe.S):
        # take the best action for the current policy


        # Find the best action by one-step lookahead

        #Update policy
        # with if, and upgrade policy

        # if policy stable shutdown

        old_action = action_values(lwe, s, V) # policy
        best_action = np.argmax(old_action)

        pi_dict[s] = {0: old_action, 1: best_action}

        if old_action != best_action:
            policy_stable = False
        policy[s] = np.eye

    if policy_stable:
        return PolicyAndValueFunction(pi_dict, V)
    else:
        return PolicyAndValueFunction(pi_dict, policy_evaluation(..., lwe))



def policy_iteration_on_line_world() -> PolicyAndValueFunction:
    """
    Creates a Line World of 7 cells (leftmost and rightmost are terminal, with -1 and 1 reward respectively)
    Launches a Policy Iteration Algorithm in order to find the Optimal Policy and its Value Function
    Returns the Policy (Pi(s,a)) and its Value Function (V(s))
    """
    # TODO
    lwe = LineWorldEnv()

    return PolicyAndValueFunction(..., ...)




def value_iteration_on_line_world() -> PolicyAndValueFunction:
    """
    Creates a Line World of 7 cells (leftmost and rightmost are terminal, with -1 and 1 reward respectively)
    Launches a Value Iteration Algorithm in order to find the Optimal Policy and its Value Function
    Returns the Policy (Pi(s,a)) and its Value Function (V(s))
    """
    # TODO
    pass


def policy_evaluation_on_grid_world() -> ValueFunction:
    """
    Creates a Grid World of 5x5 cells (upper rightmost and lower rightmost are terminal, with -1 and 1 reward respectively)
    Launches a Policy Evaluation Algorithm in order to find the Value Function of a uniform random policy
    Returns the Value function (V(s)) of this policy
    """
    # TODO
    pass


def policy_iteration_on_grid_world() -> PolicyAndValueFunction:
    """
    Creates a Grid World of 5x5 cells (upper rightmost and lower rightmost are terminal, with -1 and 1 reward respectively)
    Launches a Policy Iteration Algorithm in order to find the Optimal Policy and its Value Function
    Returns the Policy (Pi(s,a)) and its Value Function (V(s))
    """
    # TODO
    pass


def value_iteration_on_grid_world() -> PolicyAndValueFunction:
    """
    Creates a Grid World of 5x5 cells (upper rightmost and lower rightmost are terminal, with -1 and 1 reward respectively)
    Launches a Value Iteration Algorithm in order to find the Optimal Policy and its Value Function
    Returns the Policy (Pi(s,a)) and its Value Function (V(s))
    """
    # TODO
    pass


def policy_evaluation_on_secret_env1() -> ValueFunction:
    """
    Creates a Secret Env1
    Launches a Policy Evaluation Algorithm in order to find the Value Function of a uniform random policy
    Returns the Value function (V(s)) of this policy
    """
    env = Env1()
    # TODO
    pass


def policy_iteration_on_secret_env1() -> PolicyAndValueFunction:
    """
    Creates a Secret Env1
    Launches a Policy Iteration Algorithm in order to find the Optimal Policy and its Value Function
    Returns the Policy (Pi(s,a)) and its Value Function (V(s))
    """
    env = Env1()
    # TODO
    pass


def value_iteration_on_secret_env1() -> PolicyAndValueFunction:
    """
    Creates a Secret Env1
    Launches a Value Iteration Algorithm in order to find the Optimal Policy and its Value Function
    Prints the Policy (Pi(s,a)) and its Value Function (V(s))
    """
    env = Env1()
    # TODO
    pass


def demo():
    print(policy_evaluation_on_line_world())
    # print(policy_iteration_on_line_world())
    # print(value_iteration_on_line_world())

    # print(policy_evaluation_on_grid_world())
    # print(policy_iteration_on_grid_world())
    # print(value_iteration_on_grid_world())
    #
    # print(policy_evaluation_on_secret_env1())
    # print(policy_iteration_on_secret_env1())
    # print(value_iteration_on_secret_env1())
