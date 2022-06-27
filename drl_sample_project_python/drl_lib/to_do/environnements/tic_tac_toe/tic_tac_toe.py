import random

class TicTacToe:
    def __init__(self):
        self.board = [[' ', ' ', ' '], [' ', ' ', ' '], [' ', ' ', ' ']]
        self.p1 = 'X'
        self.p2 = 'O'

    def random_play(self) -> int:
         return int(random.uniform(1, 10))

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
            print("The case is already take")
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

    def chose_case(self, p):
        if p == self.p2:
            return self.random_play()
        else:
            return int(input("case 1 - 9: "))
    def play(self):
        p = self.p1
        while not self.victory():
            x = self.chose_case(p)

            if x > 0 and x < 10:
                if self.action(p, x-1):
                    p = self.change_player(p)
            else:
                print("Input case between 1 - 9")
            self.print_board()

        print(self.change_player(p) + " is win")