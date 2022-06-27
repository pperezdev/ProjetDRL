import numpy as np


class Board:
    def __init__(self, board: np.array = None):
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

    def availablePositions(self):
        positions = []
        for i in range(self.size):
            for j in range(self.size):
                if self.board[i, j] == 0:
                    positions.append([i, j])
        return positions

    def is_winner(self):
        # row
        for i in range(self.size):
            if sum(self.board[i, :]) == 3:
                self.isEnd = True
                return 1
            if sum(self.board[i, :]) == -3:
                self.isEnd = True
                return -1

        # col
        for i in range(self.size):
            if sum(self.board[:, i]) == 3:
                self.isEnd = True
                return 1
            if sum(self.board[:, i]) == -3:
                self.isEnd = True
                return -1

        # diagonal
        diag_sum1 = sum([self.board[i, i] for i in range(self.size)])
        diag_sum2 = sum([self.board[i, self.size - i - 1] for i in range(self.size)])
        diag_sum = max(abs(diag_sum1), abs(diag_sum2))
        if diag_sum == 3:
            self.isEnd = True
            if diag_sum1 == 3 or diag_sum2 == 3:
                return 1
            else:
                return -1

        # pas possible de jouer
        if (len(self.availablePositions())) == 0:
            self.isEnd = True
            return 0

        # not isEnd
        self.isEnd = False
        return None

    def hash(self):
        result = 0
        for i in range(self.size):
            for j in range(self.size):
                result *= 3
                result += self.board[i][j] % 3
        return result

    def __repr__(self) -> str:
        '''
        Renvoie le tableau du Tic Tac Toe dans une représentation lisible par l'homme en utilisantla forme suivante :
        (les indices sont remplacés par des 'X', 'O' et des les espaces pour les cellules vides) :
        '''
        result = ''
        mapping = [' ', 'X', 'O']
        for i in range(self.size):
            for j in range(self.size):
                result += ' {} '.format(mapping[self.board[i][j]])
                if j != self.size - 1:
                    result += '|'
                else:
                    result += '\n'
            if i != self.size - 1:
                result += ('-' * (2 + self.size * self.size)) + '\n'
        return result


class TicTacToeEnv:
    """
    TicTacToe est un environnement d'apprentissage par renforcement pour ce jeu, qui réagit aux déplacements des joueurs,
    met à jour l'état interne (plateau) et échantillonne les la récompense
    """

    def __init__(self):
        self.create_board()

    def create_board(self):
        self.board: Board = Board()

    def step(self, action):
        over, _ = self.board.is_winner()
        self.board.take_turn(action)
        over, winner = self.board.is_winner()
        return winner, self.board, over

    def __repr__(self):
        return self.board.__repr__()

    def reset(self):
        self.create_board()



class Player:
    pass

