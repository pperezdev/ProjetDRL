from ..do_not_touch.result_structures import PolicyAndActionValueFunction
from ..do_not_touch.single_agent_env_wrapper import Env3

class morpion:
    def __init__(self):
        self.board = [[' ', ' ', ' '], [' ', ' ', ' '], [' ', ' ', ' ']]
        self.p1 = '🍪'
        self.p2 = '💓'

    def print_board(self):
        print("-------------")
        for i in range(3):
            print(f"| {self.board[0][i]} | {self.board[1][i]} | {self.board[2][i]} |")
            print("-------------")

    def action(self, p, x) -> bool:
        y = 0
        if x < 3:
            y = 0
        else:

            y = int(abs(x / 3))
            if y == 1:
                x = x - 3
            else:
                x = int(x - (3 * y))
        print(x)
        print(y)
        if self.board[x][y] == ' ':
            self.board[x][y] = p
            return True
        else:
            print("The case is take")
            return False

    def victory(self) -> bool:
        for i in range(3):
            if (self.board[i][0] == self.board[i][1] == self.board[i][2]) and self.board[i][0] != ' ':
                return True
        for i in range(3):
            if (self.board[0][i] == self.board[1][i] == self.board[2][i]) and self.board[0][i] != ' ':
                return True
        if (self.board[0][0] == self.board[1][1] == self.board[2][2]) and self.board[1][1] != ' ':
            return True
        if (self.board[0][2] == self.board[1][1] == self.board[2][0]) and self.board[1][1] != ' ':
            return True
        return False

    def change_player(self, p):
        if p == self.p1:
            return self.p2
        else:
            return self.p1

    def play(self):
        p = self.p1
        while not self.victory():
            x = int(input("case 1 - 9: "))

            if x > 0 and x < 10:
                if self.action(p, x-1):
                    p = self.change_player(p)
            else:
                print("Input case between 1 - 9")
            self.print_board()

        print(self.change_player(p) + " is win")

def sarsa_on_tic_tac_toe_solo() -> PolicyAndActionValueFunction:
    """
    Creates a TicTacToe Solo environment (Single player versus Uniform Random Opponent)
    Launches a SARSA Algorithm in order to find the optimal epsilon-greedy Policy and its action-value function
    Returns the optimal epsilon-greedy Policy and its Action-Value function (Q(s,a))
    Experiment with different values of hyper parameters and choose the most appropriate combination
    """
    # TODO
    m = morpion()
    m.play()
    return 0



def q_learning_on_tic_tac_toe_solo() -> PolicyAndActionValueFunction:
    """
    Creates a TicTacToe Solo environment (Single player versus Uniform Random Opponent)
    Launches a Q-Learning algorithm in order to find the optimal greedy Policy and its action-value function
    Returns the optimal greedy Policy and its Action-Value function (Q(s,a))
    Experiment with different values of hyper parameters and choose the most appropriate combination
    """



    # TODO
    pass


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
