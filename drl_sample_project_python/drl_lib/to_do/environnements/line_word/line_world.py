import numpy as np

class LineWorldEnv:
    def __init__(self):
        self.S = [0, 1, 2, 3, 4, 5, 6]
        self.A = [0, 1]
        self.R = [-1.0, 0.0, 1.0]

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


def action_value_function(lwe, V, pi, s, a, gamma):
    q = 0
    total = 0.0
    for s_p in lwe.S:
        for r in range(len(lwe.R)):
            total += lwe.p[s, a, s_p, r] * (lwe.R[r] + gamma * V[s_p])
    total *= pi[s, a]
    q += total

    return q

# Policy Evaluation
def policy_evaluation(lwe, pi, V, gamma, theta):
    V_updated = np.copy(V)
    improved = True

    while True:
        delta = 0
        for s in range(len(lwe.S)):
            V_new = 0

            for a in range(len(lwe.A)):
                prob_a = pi[s][a]
                q_s_a = action_value_function(lwe, V, pi, s, a, gamma)

                V_new += prob_a * q_s_a

            delta = max(delta, np.abs(V_new - V_updated[s]))
            V_updated[s] = V_new

        if delta < theta:
            break

    if np.any(V != V_updated):
        improved = False
    # if np.array_equal(V, V_updated):
        # improved = False

    return V_updated, improved

def policy_improvement(lwe, pi, V, gamma):
    for s in lwe.S:
        q_s = np.zeros([len(lwe.A)],)

        for a in lwe.A:
            q_s[a] = action_value_function(lwe, V, pi, s, a, gamma)

        best_action = np.argmax(q_s)
        pi[s] = np.eye(len(lwe.A))[best_action]
    return pi

def policy_iteration(lwe, pi, V, gamma, theta):
    k = 0
    while True:
        k += 1

        V, is_stable = policy_evaluation(lwe, pi, V, gamma, theta)
        pi = policy_improvement(lwe, pi, V, gamma)
        if is_stable is False:
            print("Finished after " + str(k) + " iterations.")
            break
    return V, pi

def Start():
    lwe = LineWorldEnv()
    pi = np.ones([len(lwe.S), len(lwe.A)]) * 0.5

    V = np.zeros([len(lwe.S), 1])
    gamma = 0.999
    theta = 0.0000001

    print("######################")
    print(policy_iteration(lwe, pi, V, gamma, theta))

Start()