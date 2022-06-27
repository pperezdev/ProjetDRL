import numpy as np



class TicTacToeEnv():

    def __init__(self, board):
        self.size: int = 3
        self.cells: np.array
        if board is not None:
            self.board = board
        else:
            self.board = np.zeros((self.size, self.size))
        self.first_player = True
        self.isEnd = False

    def take_turn(self, cell):
        player_identifier = 1
        if not self.first_player:
            player_identifier = -1
        self.board[cell] = player_identifier
        # changement du jouer apres jouer
        self.first_player = not self.first_player

    def is_possible(self, action):
        return self.board[action] == 0

    def possible_actions(self):
        return np.array([(i,j)
                         for i in range(BOARD_ROWS)
                         for j in range(BOARD_COLS)
                         if self.is_possible((i,j))])

    def winner(self):
        # row
        for i in range(BOARD_ROWS):
            if sum(self.board[i, : ]) == 3:
                self.isEnd = True
                return 1
            if sum(self.board[i, : ]) == -3:
                self.isEnd = True
                return -1

        # col
        for i in range(BOARD_COLS):
            if sum(self.board[:, i]) == 3:
                self.isEnd = True
                return 1
            if sum(self.board[:, i]) == -3:
                self.isEnd = True
                return -1

        # diagonal
        diag_sum1 = sum([self.board[i, i] for i in range(BOARD_COLS)])
        diag_sum2 = sum([self.board[i, BOARD_COLS - i - 1] for i in range(BOARD_COLS)])
        diag_sum = max(abs(diag_sum1), abs(diag_sum2))
        if diag_sum == 3:
            self.isEnd = True
            if diag_sum1 == 3 or diag_sum2 == 3:
                return 1
            else:
                return -1

        # pas possible de jouer
        if(len(self.availablePositions())) == 0:
            self.isEnd = True
            return  0

        #not isEnd
        self.isEnd = False
        return None

    def availablePositions(self):
        positions = []
        for i in range(BOARD_ROWS):
            for j in range(BOARD_COLS):
                if self.board[i, j] == 0:
                    positions.append([i, j])
        return positions

    def hash(self):
        result = 0
        for i in range(BOARD_ROWS):
            for j in range(BOARD_COLS):
                result *



