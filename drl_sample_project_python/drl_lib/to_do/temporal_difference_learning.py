from ..do_not_touch.result_structures import PolicyAndActionValueFunction
from ..do_not_touch.single_agent_env_wrapper import Env3
from drl_sample_project_python.drl_lib.to_do.environnements.tic_tac_toe.tic_tac_toe_env_test import TicTacToeEnv
import numpy as np
import random

total_episodes = 25000        # Total number of training episodes
total_test_episodes = 200     # Total number of test episodes
max_steps = 300               # Max steps per episode

learning_rate = 0.01          # Learning rate
gamma = 0.99                  # Discounting rate

# Exploration parameters
epsilon = 1.0                 # Exploration rate
max_epsilon = 1.0             # Exploration probability at start
min_epsilon = 0.001           # Minimum exploration probability
decay_rate = 0.01             # Exponential decay rate for exploration proba


def sarsa_on_tic_tac_toe_solo() -> PolicyAndActionValueFunction:
    """
    Creates a TicTacToe Solo environment (Single player versus Uniform Random Opponent)
    Launches a SARSA Algorithm in order to find the optimal epsilon-greedy Policy and its action-value function
    Returns the optimal epsilon-greedy Policy and its Action-Value function (Q(s,a))
    Experiment with different values of hyper parameters and choose the most appropriate combination
    """
    # TODO
    m = TicTacToeEnv()




def q_learning_on_tic_tac_toe_solo() -> PolicyAndActionValueFunction:
    """
    Creates a TicTacToe Solo environment (Single player versus Uniform Random Opponent)
    Launches a Q-Learning algorithm in order to find the optimal greedy Policy and its action-value function
    Returns the optimal greedy Policy and its Action-Value function (Q(s,a))
    Experiment with different values of hyper parameters and choose the most appropriate combination
    """
    env = TicTacToeEnv()
    qtable = np.zeros((9,681))

    def greddy_policy(Q, state):
        # Exploitation
        if (random.uniform(0, 1) > epsilon):
            action = np.argmax(Q[state])

        # Exploration
        else:
            action = env.step() # random action
        return action

    for episode in range(total_episodes):  # episode

        # Reset environnement
        state = env.reset()
        step = 0
        final = False

        # Reduce epsilon
        epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay_rate * episode)

        for step in range(max_steps):  # step

            action = greddy_policy(qtable, state)

            # Take action (a) and get reward (r), transit to next state (s)
            new_state, reward, done, info = env.step(action)

            # TD update Q(s,a):= Q(s,a) + lr [R(s,a) + gamma * max Q(s',a') - Q(s,a)]
            qtable[state][action] = qtable[state][action] + learning_rate * (reward + gamma *
                                                                             np.max(qtable[new_state] - qtable[state][
                                                                                 action]))

            if final == True:
                break

            state = new_state



def expected_sarsa_on_tic_tac_toe_solo() -> PolicyAndActionValueFunction:
    """
    Creates a TicTacToe Solo environment (Single player versus Uniform Random Opponent)
    Launches a Expected SARSA Algorithm in order to find the optimal epsilon-greedy Policy and its action-value function
    Returns the optimal epsilon-greedy Policy and its Action-Value function (Q(s,a))
    Experiment with different values of hyper parameters and choose the most appropriate combination
    """
    # TODO
    pass


def sarsa_on_secret_env3() -> PolicyAndActionValueFunction:
    """
    Creates a Secret Env3
    Launches a SARSA Algorithm in order to find the optimal epsilon-greedy Policy and its action-value function
    Returns the optimal epsilon-greedy Policy and its Action-Value function (Q(s,a))
    Experiment with different values of hyper parameters and choose the most appropriate combination
    """
    env = Env3()
    # TODO
    pass


def q_learning_on_secret_env3() -> PolicyAndActionValueFunction:
    """
    Creates a Secret Env3
    Launches a Q-Learning algorithm in order to find the optimal greedy Policy and its action-value function
    Returns the optimal greedy Policy and its Action-Value function (Q(s,a))
    Experiment with different values of hyper parameters and choose the most appropriate combination
    """
    env = Env3()
    # TODO
    pass


def expected_sarsa_on_secret_env3() -> PolicyAndActionValueFunction:
    """
    Creates a Secret Env3
    Launches a Expected SARSA Algorithm in order to find the optimal epsilon-greedy Policy and its action-value function
    Returns the optimal epsilon-greedy Policy and its Action-Value function (Q(s,a))
    Experiment with different values of hyper parameters and choose the most appropriate combination
    """
    env = Env3()
    # TODO
    pass


def demo():
    print(sarsa_on_tic_tac_toe_solo())
    print(q_learning_on_tic_tac_toe_solo())
    print(expected_sarsa_on_tic_tac_toe_solo())

    print(sarsa_on_secret_env3())
    print(q_learning_on_secret_env3())
    print(expected_sarsa_on_secret_env3())
