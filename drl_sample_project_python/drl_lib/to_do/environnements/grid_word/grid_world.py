from drl_sample_project_python.drl_lib.do_not_touch.mdp_env_wrapper import Env1
from drl_sample_project_python.drl_lib.do_not_touch.result_structures import ValueFunction, PolicyAndValueFunction
import numpy as np

class GridWorld:
    def __init__(self, width, height, start):
        self.width = width
        self.height = height
        self.i = start[0]
        self.j = start[1]

    def set(self, rewards, actions):
        """
        rewards dict: (row,col): reward
		actions dict: (row,col): liste d'action
        """
        self.rewards = rewards
        self.actions = actions

    def set_state(self, s):
        self.i = s[0]
        self.j = s[1]

    def current_state(self):
        return (self.i, self.j)

    def is_terminal(self, state):
        return state not in self.actions

    def game_over(self):
        return (self.i, self.j) not in self.actions

    def move(self, action):
        """
        Check if a action is possible
        action possible:
        D -> Down
        U -> Up
        L -> Left
        R -> Right
        """
        if action in self.actions[self.i, self.j]:
            if action == 'U':
                self.i +=1
            elif action == "D":
                self.i -=1
            elif action == "R":
                self.j +=1
            elif action == "L":
                self.j -=1

        return self.rewards(self.i, self.j)

    def all_states(self):
        """
        Donne une list complete des actions possibles dans le grid world
        """
        return set(list(self.actions.keys()) + list(self.rewards.keys()))


def grid():
    grd = GridWorld(5, 5, (2, 0))
    rewards = {(0,4):1, (1,4):-1}
    actions = {
        (0,0): ('D', 'R'),
        (0,1): ('D', 'R', 'L'),
        (0, 2): ('R', 'L'),
        (0, 3): ('D', 'R', 'L'),

        (1,0): ('U', 'D', 'R'),
        (1, 1): ('U', 'D', 'L'),
        (1, 3): ('U', 'D', 'R'),

        (2, 0): ('U', 'D', 'R'),
        (2, 1): ('U', 'D', 'L', 'R'),
        (2, 2): ('D', 'L', 'R'),
        (2, 3): ('U', 'D', 'L', 'R'),
        (2, 4): ('U', 'D', 'L'),

        (3, 0): ('U', 'D', 'R'),
        (3, 1): ('U', 'D', 'L', 'R'),
        (3, 2): ('U', 'L', 'R'),
        (3, 3): ('U', 'D', 'L', 'R'),
        (3, 4): ('U', 'D', 'L'),

        (4, 0): ('U', 'R'),
        (4, 1): ('U', 'L'),
        (4, 3): ('U', 'R'),
        (4, 4): ('U', 'L')
    }

    grd.set(rewards, actions)
    return grd

# ---------------------------------------------------------------------------------------------------------------------
#                                                Grid World
# ---------------------------------------------------------------------------------------------------------------------


ALL_POSSIBLE_ACTIONS = ('U','D','L','R')
GAMMA = 1
thresold = 10e-4



def policy_evaluation_on_grid_world(V, gamma) -> ValueFunction:
    """
    Creates a Grid World of 5x5 cells (upper rightmost and lower rightmost are terminal, with -1 and 1 reward respectively)
    Launches a Policy Evaluation Algorithm in order to find the Value Function of a uniform random policy
    Returns the Value function (V(s)) of this policy
    """
    # TODO
    print("Grid World")
    create_grid = grid()
    states = create_grid.all_states()

    while True:
        delta = 0
        for s in states:
            old_v = V[s]
            if s in create_grid.actions:
                total = 0.0
                p_a = 1.0 / len(create_grid.actions[s])  # each actions has equal prob since uniform
                for sp in create_grid.actions[s]:
                    create_grid.set_state(s)
                    r = create_grid.move(sp)
                    total += p_a * (r + gamma * V[create_grid.current_state()])
                V[s] = total
                delta = max(delta, np.abs(old_v - V[s]))
        if delta < thresold:
            break
    return V


def policy_iteration_on_grid_world() -> PolicyAndValueFunction:
    """
    Creates a Grid World of 5x5 cells (upper rightmost and lower rightmost are terminal, with -1 and 1 reward respectively)
    Launches a Policy Iteration Algorithm in order to find the Optimal Policy and its Value Function
    Returns the Policy (Pi(s,a)) and its Value Function (V(s))
    """

    create_grid = grid()
    print('rewards', create_grid.rewards)

    policy = {}
    for s in create_grid.actions.keys():
        policy[s] = np.random.choice(ALL_POSSIBLE_ACTIONS)

    print('initial policy', policy)

    # initialize V(s)
    V = {}
    states = create_grid.all_states()
    for s in states:
        if s in create_grid.actions:
            V[s] = np.random.random()
        else:
            V[s] = 0

    while True:
        # policy evaluation
        policy_evaluation_on_grid_world(V, GAMMA)

        #policy improvement step
        is_policy_converged = True
        for s in states:
            if s in policy:
                old_a = policy[s]
                new_a = None
                best_value = float('-inf')
                for i in ALL_POSSIBLE_ACTIONS:
                    v = 0
                    for j in ALL_POSSIBLE_ACTIONS:
                        if i == j:
                            p = 0.5
                        else:
                            p = 0.5 / 3
                        create_grid.set_state(i)
                        r = create_grid.move(j)
                        v += p * (r + GAMMA * V[create_grid.current_state()])
                    if v > best_value:
                        best_value = v
                        new_a = i
                policy[s] = new_a
                if new_a != old_a:
                    is_policy_converged = False
        if is_policy_converged:
            break
    return V


def value_iteration_on_grid_world() -> PolicyAndValueFunction:
    """
    Creates a Grid World of 5x5 cells (upper rightmost and lower rightmost are terminal, with -1 and 1 reward respectively)
    Launches a Value Iteration Algorithm in order to find the Optimal Policy and its Value Function
    Returns the Policy (Pi(s,a)) and its Value Function (V(s))
    """
    # TODO
    create_grid = grid()

    #randomly instantiate
    policy = {}
    for s in create_grid.actions.keys():
        policy[s] = np.random.choice(ALL_POSSIBLE_ACTIONS)

    #initialize V(s) randomly between 0 and 1
    V = {}
    states = create_grid.all_states()
    for s in states:
        if s in create_grid.actions:
            V[s] = np.random.random()
        else:
            #terminal state so we set Value to 0
            V[s] = 0

    while True:
        max_change = 0
        for s in states:
            old_vs = V[s]
            if s in policy:
                new_v = float('-inf')
                for i in ALL_POSSIBLE_ACTIONS:
                    create_grid.set_state(s)
                    r = create_grid.move(i)
                    v = r + GAMMA * V[create_grid.current_state()]
                    if v > new_v:
                        new_v = v
                V[s] = new_v
                max_change = max(max_change, np.abs(old_vs - V[s]))
        if max_change < thresold:
            break
    for s in policy.keys():
        best_act = None
        best_value = float('-inf')
        for a in ALL_POSSIBLE_ACTIONS:
            create_grid.set_state(a)
            r = create_grid.move(a)
            v = r + GAMMA * V[create_grid.current_state()]
            if v > best_value:
                best_value = v
                best_act = a
        policy[s] = best_act
    return policy