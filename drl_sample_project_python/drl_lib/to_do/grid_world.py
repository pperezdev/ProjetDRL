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


