import numpy as np

BOARD_ROWS, BOARD_COLS = 3, 3


class TicTacToeEnv():

    def __init__(self, p1, p2):
        self.board = np.zeros((BOARD_ROWS, BOARD_COLS))
        self.p1 = p1
        self.p2 = p2


