from drl_sample_project_python.drl_lib.do_not_touch.result_structures import PolicyAndActionValueFunction
from drl_sample_project_python.drl_lib.do_not_touch.single_agent_env_wrapper import Env3
from drl_sample_project_python.drl_lib.to_do.environnements.tic_tac_toe.tic_tac_toe_env_test import Player, TicTacToeEnv
from drl_sample_project_python.drl_lib.to_do.environnements.tic_tac_toe.tic_tac_toe import TicTacToe

import numpy as np
import random


class QLearner(Player):
    def __init__(self, epsilon=0.2, alpha=0.3, gamma=0.9):
        self.q = {}
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma

    def getQ(self, state, action):
        if self.q.get([state, action]) is None:
            self.q[(state, action)] = 1.0
        return self.q.get([state, action])

    def move(self, board):
        self.last_board = tuple(board)
        actions = self.available_moves(board)

        # explore
        if random.random() < self.epsilon:
            self.last_move = random.choice(actions)
            return self.last_move

        qs = ([self.getQ(self.last_board, a) for a in actions])
        maxQ = max(qs)

        # Exploitation
        if qs.count(maxQ) > 1:
            # more than 1 best option; choose among them randomly
            best_options = [i for i in range(len(actions)) if qs[i] == maxQ]
            i = random.choice(best_options)
        else:
            i = qs.index(maxQ)

        self.last_move = actions[i]
        return actions[i]

    def reward(self, value, board):
        """
        Take action (a) and get reward (r), transit to next state (s)
        """
        if self.last_move:
            self.learn(self.last_board, self.last_move, value, tuple(board))

    def learn(self, state, action, reward, result_state):
        """
        TD update Q(s,a):= Q(s,a) + lr [R(s,a) + gamma * max Q(s',a') - Q(s,a)]
        """
        prev = self.getQ(state, action)
        maxqnew = max([self.getQ(result_state, a) for a in self.available_moves(state)])
        self.q[(state, action)] = prev + self.alpha * ((reward + self.gamma * maxqnew) - prev)


def q_learning_on_tic_tac_toe_solo() -> PolicyAndActionValueFunction:
    """
    Creates a TicTacToe Solo environment (Single player versus Uniform Random Opponent)
    Launches a Q-Learning algorithm in order to find the optimal greedy Policy and its action-value function
    Returns the optimal greedy Policy and its Action-Value function (Q(s,a))
    Experiment with different values of hyper parameters and choose the most appropriate combination
    """
    p1 = QLearner()





def demo():
    print(q_learning_on_tic_tac_toe_solo())
