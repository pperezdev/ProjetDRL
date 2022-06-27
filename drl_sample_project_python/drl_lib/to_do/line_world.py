import numpy as np

class LineWorldEnv:
    def __init__(self):
        self.S = [0, 1, 2, 3, 4, 5, 6]
        self.A = [0, 1]
        self.R = [-1.0, 0.0, 1.]

        self.p = np.zeros((len(self.S), len(self.A), len(self.S), len(self.R)))

        for s in self.S[1:-1]:
            if s == 1:
                self.p[s, 0, s - 1, 0] = 1.0
            else:
                self.p[s, 0, s - 1, 1] = 1.0

            if s == 5:
                self.p[s, 1, s + 1, 2] = 1.0
            else:
                self.p[s, 1, s + 1, 1] = 1.0

def action_value_funtion(lwe, V, s, a, gamma):
    q = 0

    for p_sp, sp, r in lwe.p[s, a]:
        print("p_sp: ", p_sp)
        print("r: ", r)
        print("gamma: ", gamma)
        print("sp: ", sp)
        print("V: ", V)
        print("V[sp]: ", V[int(sp)])
        q += p_sp * (r + gamma * V[int(sp)])

    return q

# Policy Evaluation
def policy_evaluation(lwe, V, gamma, theta):
    while True:
        delta = 0

        for s in lwe.S:
            v = 0
            # print("s: ", s)
            for a in lwe.A:
                # print("a: ", a)
                q_s_a = action_value_funtion(lwe, V, s, a, gamma)
                # print("q_s_a: ", q_s_a)
                # print("pi[s, a]: ", pi[s, a])
                v += pi[s, a] * q_s_a
                # print("v: ", v)
                # print("############")

            delta = max(delta, np.abs(v - V[s]))
            V[s] = v

        if delta < theta:
            break

    return V

def policy_improvement(lwe, V, gamma):
    for s in lwe.S:
        q_s = np.zeros((lwe.A),)

        for a in lwe.A:
            # print("V: ", V)
            # print("s: ", s)
            # print("a: ", a)
            # print("gamma: ", gamma)
            q_s[a] = action_value_funtion(lwe, V, s, a, gamma)

        best_action = np.argmax(q_s)
        pi[s] = np.eye(lwe.A)[best_action]
    return pi

    #lwe = LineWorldEnv()
    #V = np.zeros((len(lwe.S)),)

    #pi = np.ones((len(lwe.S), len(lwe.A))) * 0.5

    #gamma = 0.999
    #theta = 0.0000001

    # print(action_value_funtion(lwe, V, 1, 0, gamma))
    # print(policy_evaluation(lwe, V, gamma, theta))
    #V = policy_evaluation(lwe, V, gamma, theta)
    #print("old V: ", V)
    #print("impove V: ", policy_improvement(lwe, V, gamma))


