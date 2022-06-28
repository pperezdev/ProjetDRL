import numpy as np

class Gridworld:
    def __init__(self, width=5, height=5, actions=4,
                 pos_X_victory=0, pos_Y_victory=0,
                 pos_X_defeat=4, pos_Y_defeat=4):

        pos_X_victory = self.condition_array_superior(pos_X_victory, width)
        pos_Y_victory = self.condition_array_superior(pos_Y_victory, height)
        pos_X_defeat = self.condition_array_superior(pos_X_defeat, width)
        pos_Y_defeat = self.condition_array_superior(pos_Y_defeat, height)

        self.width = width
        self.height = height

        self.S = np.arange(width * height).reshape((width, height))
        self.A = np.arange(actions)
        self.R = np.array([-1.0, 0.0, 1.0])

        self.p = np.zeros((len(self.S), len(self.A), len(self.S), len(self.R)))

        # for s in self.S[1:][:1]:
        #    pass

        #self.p[pos_X_victory, pos_Y_victory] = -1
        #self.p[pos_X_defeat, pos_Y_defeat] = 1

    def __repr__(self):
        return str({
            "heigh": self.height,
            "width": self.width,
            "S": self.S,
            "A": self.A,
            "R": self.R,
            "p": self.p
        })

    def condition_array_superior(self, pos, size):
        if pos >= size:
            pos = size - 1
        else:
            pos = self.condition_array_inferior_zero(pos)
        return pos

    def condition_array_inferior_zero(self, pos):
        if pos < 0:
            pos = 0
        return pos


def action_value_funtion(lwe, V, s, a, gamma):
    q = 0

    for (p_sp, sp, r) in lwe.p[s, a]:
        # print("p_sp: ", p_sp)
        # print("r: ", r)
        # print("gamma: ", gamma)
        # print("sp: ", sp)
        # print("V: ", V)
        # print("V[sp]: ", V[int(sp)])
        q += p_sp * (r + gamma * V[int(sp)])

    return q

# Policy Evaluation
def compute_q_value_for_s_a(gwe, V, s, a, gamma):
    q = 0

    for (p_sPrime, sPrime, r_ss_a, done) in gwe.P[s][a]:
        q += p_sPrime * (r_ss_a + gamma * V[sPrime])

    return q

def policy_evaluation(gwe, pi, V, gamma, theta):
    i = 0
    while True:
        i += 1
        delta = 0

        for s in range(gwe.S):
            V_new = 0
            for a in range(gwe.A):
                prob_a = pi[s][a]
                q_s_a = compute_q_value_for_s_a(gwe, V, s, a, gamma)

                V_new += prob_a * q_s_a

            delta = max(delta, np.abs(V_new - V[s]))
            V[s] = V_new

        if (delta < theta):
            print("Terminé après " + str(i) + " itérations")
    return V

gwe = Gridworld()

#print(repr(gw))

print(gwe.S)
print("##############")
pi = np.ones([len(gwe.S), len(gwe.A)])  * 0.25

V = np.zeros([len(gwe.S), 1])
gamma = 1.0
theta = 0.00001
print(policy_evaluation(gwe, pi, V, gamma, theta))

# print(gw.S[0][1:])
# print(gw.S[0][0])

