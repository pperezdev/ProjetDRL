import copy
import random

from ..do_not_touch.mdp_env_wrapper import Env1
from ..do_not_touch.result_structures import ValueFunction, PolicyAndValueFunction
from line_world import LineWorldEnv
import numpy as np
try:
    from grid_world import grid
    print("&")
except:
    pass


########## TO-DO
"""
def action_values
def policy_improvement
def policy_iteration_on_line_world
def value_iteration_on_line_world
def
def
def
def
def
"""

########## DONE ##########
def policy_evaluation_on_line_world(pi: np.ndarray, lwe: LineWorldEnv, theta=0.0000001):
    """
        Creates a Line World of 7 cells (leftmost and rightmost are terminal, with -1 and 1 reward respectively)
        Launches a Policy Evaluation Algorithm in order to find the Value Function of a uniform random policy
        Returns the Value function (V(s)) of this policy
    """
    """
    _input:
        π: Stratégie (policy)

    _params: theta > 0

    """

    # Initialisation de V(s)
    V = np.random.random((len(lwe.S),))
    gamma = 0.999
    # Exception des V terminaux
    V[0] = 0.0
    V[6] = 0.0

    while True:
        delta = 0
        for s in lwe.S:
            v = V[s]
            V[s] = 0.0
            for a in lwe.A:
                total = 0.0
                for sp in lwe.S:
                    for r in range(len(lwe.R)):
                        total += lwe.p[sp, a, s, r] * (lwe.R[r] + gamma * V[sp])
                        pass
                total *= pi[s, a]
                V[s] += total
            delta = max(delta, abs(v - V[s]))
        if delta < theta:
            break

    return V

def action_values(lwe: LineWorldEnv, state, V, gamma=1):
    B = np.zeros(len(lwe.A))
    for i in lwe.A:  # 0, 1
        print(f"==> is is: {i}")
        for probability, next_state, reward in lwe.p[state][i]:
            print(f"==> je suis la {lwe.p[state][i]}")
            B[i] += probability * (reward + gamma * V[next_state])
    return B

def policy_improvement(lwe: LineWorldEnv, V):
    policy = np.ones((len(lwe.S), len(lwe.A))) * 1 / (len(lwe.S) - len(lwe.A))
    policy_stable = True

    for s in lwe.S:
        pi_dict = dict()
        old_action = []
        for a in lwe.A:
            # Find the best action by one-step lookahead
            old_action.append(action_values(lwe, s, V))  # policy

        # take the best action for the current policy
        best_action = np.argmax(old_action)

        # Update policy
        # with if, and upgrade policy

        # if policy stable shutdown

        # pi_dict[s] = {0: old_action, 1: best_action}

        if old_action != best_action:
            policy_stable = False
        policy[s] = np.sum([np.eye(lwe.A)[i] for i in best_action], axis=0) / len(best_action)
    return policy

    # if policy_stable:
    #    return PolicyAndValueFunction(pi_dict, V)
    # else:
    #    return PolicyAndValueFunction(pi_dict, policy_evaluation(..., lwe))


def policy_iteration_on_line_world(pi: np.ndarray, lwe: LineWorldEnv, theta=0.0000001) -> PolicyAndValueFunction:
    """
    Creates a Line World of 7 cells (leftmost and rightmost are terminal, with -1 and 1 reward respectively)
    Launches a Policy Iteration Algorithm in order to find the Optimal Policy and its Value Function
    Returns the Policy (Pi(s,a)) and its Value Function (V(s))
    """


    # Initialisation de V(s)
    V = np.random.random((len(lwe.S),))
    gamma = 0.999
    # Exception des V terminaux
    V[0] = 0.0
    V[6] = 0.0

    # Evaluation de la stratégie
    while True:
        delta = 0
        for s in lwe.S:
            v = V[s]
            # V[s] = 0.0
            for a in lwe.A:
                total = 0.0
                for sp in lwe.S:
                    for r in range(len(lwe.R)):
                        # total += lwe.p[sp, a, s, r] * (lwe.R[r] + gamma * V[sp])
                        # print(
                            # f"lwe.R[r] ({type(lwe.R[r])}): {lwe.R[r]}",
                            # f"gamma ({type(gamma)}): {gamma}",
                            # f"V[sp] ({type(V[sp])}): {V[sp]}",
                            # sep="\n"
                        # )
                        # print("###############################")
                        # print(
                            # f"sp ({type(sp)}): {sp}",
                            # f"pi ({type(pi)}): {pi}",
                            # f"pi[s] ({type(pi[s])}): {pi[s]}",
                            # f"pi[s, a] ({type(pi[s, a])}): {pi[s, a]}",
                            # f"s ({type(s)}): {s}",
                            # f"s ({type(r)}): {r}",
                            # sep="\n"
                        # )

                        # print(f"p(s',r|s,a) ({type(lwe.p[sp, int(pi[s, a]), s, r])}): {lwe.p[sp, int(pi[s, a]), s, r]}")

                        total += lwe.p[sp, int(pi[s, a]), s, r] * (lwe.R[r] + gamma * V[sp])
                        # print("total: ",total)
                        pass
                # total *= pi[s, a]
                V[s] = total
                # print("V[s]: ", V[s])
            delta = max(delta, abs(v - V[s]))
        if delta < theta:
            break

    # Amélioration de la stratégie
    policy_stable = True
    for s in lwe.S:
        # old_action = pi[s]
        old_action = V[s]

        ###################

        for a in lwe.A:
            total = 0.0
            for sp in lwe.S:
                for r in range(len(lwe.R)):
                    total += lwe.p[sp, a, s, r] * (lwe.R[r] + gamma * V[sp])
                    pass
                #total *= pi[s, a]

                new_action = np.argmax(total)

            ###################
            """
            #for a in lwe.A:
            for sp in lwe.S:
                for r in range(len(lwe.R)):
                    total += lwe.p[sp, V, s, r] * (lwe.R[r] + gamma * V[sp])
            new_action = np.argmax(total)
            """
            # print("old_action : ", old_action, end="\n")
            # print("new_action : ", new_action)
            if old_action != new_action:
                policy_stable = False

        if policy_stable:
            break

    return V

def value_iteration_on_line_world() -> PolicyAndValueFunction:
    """
    Creates a Line World of 7 cells (leftmost and rightmost are terminal, with -1 and 1 reward respectively)
    Launches a Value Iteration Algorithm in order to find the Optimal Policy and its Value Function
    Returns the Policy (Pi(s,a)) and its Value Function (V(s))
    """
    lwe = LineWorldEnv()
    V = {s: 0 for s in lwe.S}

    while True:
        old_V = V.copy()

        for s in lwe.S:
            Q = {}
            for a in lwe.A:
                Q[a] = lwe.p * (lwe.R * old_V[s+1])  # self.p = np.zeros((len(self.S), len(self.A), len(self.S), len(self.R)))
                print("######")
                print(Q[a])
                print("######")

            V[s] = max(Q.values())

        if all(old_V[s] == V[s] for s in lwe.S):
            break

        return V


    """
    lwe = LineWorldEnv()
    theta = 1e-8

    def one_step(state, V):
        A = np.zeros(lwe.A)
        for a in range(len(lwe.A)):
            for prob, next_state, reward in lwe.p[state][a]:
                A[a] += prob * (reward + 1 * V[next_state])
    V = np.zeros(lwe.S)
    while True:
        delta = 0
        for s in range(len(lwe.S)):
            A = one_step(s, V)
            best_state = np.max(A)
            delta = max(delta, np.abs(best_state - V[s]))
            V[s] = best_state
        if delta<theta:
            break
    policy = policy_improvement(lwe, V)
    return policy, V
    """


def policy_evaluation_on_grid_world() -> ValueFunction:
    """
    Creates a Grid World of 5x5 cells (upper rightmost and lower rightmost are terminal, with -1 and 1 reward respectively)
    Launches a Policy Evaluation Algorithm in order to find the Value Function of a uniform random policy
    Returns the Value function (V(s)) of this policy
    """
    # TODO
    print("Grid World")
    create_grid = grid()

    print(f"rewards : {create_grid.rewards} ")




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


def demo_test():
    right_pi = np.zeros((len(LineWorldEnv().S), len(LineWorldEnv().A)))
    right_pi[:, 1] = 1.0

    left_pi = np.zeros((len(LineWorldEnv().S), len(LineWorldEnv().A)))
    left_pi[:, 0] = 1.0

    random_pi = np.random.random((len(LineWorldEnv().S), len(LineWorldEnv().A))) #* 0.5

    #random_pi = np.zeros((len(LineWorldEnv().S), len(LineWorldEnv().A)))  # * 0.5
    random_pi = np.ones((len(LineWorldEnv().S), len(LineWorldEnv().A)))  # * 0.5

    random_pi = np.random.random((len(LineWorldEnv().S), len(LineWorldEnv().A)))

    random_pi = np.random.random_integers(0, high=1, size=(len(LineWorldEnv().S), len(LineWorldEnv().A)))

    # print(f"Stratégie tout le temps aller à droite: ", policy_evaluation_on_line_world(pi=right_pi, lwe=LineWorldEnv()))
    # print(f"Stratégie tout le temps aller à gauche: ", policy_evaluation_on_line_world(pi=left_pi, lwe=LineWorldEnv()))
    # print(f"Stratégie aléatoire: ", policy_evaluation_on_line_world(pi=random_pi, lwe=LineWorldEnv()))

    # test_pi = np.ones((len(LineWorldEnv().S), len(LineWorldEnv().A)))
    # print(f"Stratégie aléatoire: ", policy_iteration_on_line_world(pi=test_pi, lwe=LineWorldEnv()))


    #print(policy_evaluation_on_line_world())

    #V = policy_evaluation_on_line_world()

    #lwe = LineWorldEnv()
    #Q = np.zeros([len(lwe.S), len(lwe.A)])
    #for i in range(len(lwe.S)):
    #    Q[i] = action_values(lwe,i, V)
    #print("Action-Value Function:")
    #print(Q)

    print(policy_iteration_on_line_world(pi=random_pi, lwe=LineWorldEnv()))

def demo():
    demo_test()
    #print(value_iteration_on_line_world())

    #print(policy_evaluation_on_grid_world())
    # print(policy_iteration_on_grid_world())
    # print(value_iteration_on_grid_world())
    #
    # print(policy_evaluation_on_secret_env1())
    # print(policy_iteration_on_secret_env1())
    # print(value_iteration_on_secret_env1())
