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
                self.p[s, 0, s - 1, @1] = 1.0

            if s == 5:
                self.p[s, 1, s + 1, 2] = 1.0
            else:
                self.p[s, 1, s + 1, 1] = 1.0

def action_value_funtion(lwe, V, s, a, gamma):
    q = 0

    for p_sp, sp, r in lwe.p[s, a]:
        # print("p_sp: ", p_sp)
        # print("r: ", r)
        # print("gamma: ", gamma)
        # print("sp: ", sp)
        # print("V: ", V)
        # print("V[sp]: ", V[int(sp)])
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
    pi = []
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

def policy_iteration_on_line_world_NEW(pi: np.ndarray, lwe: LineWorldEnv, theta=0.0000001):
    """
    Creates a Line World of 7 cells (leftmost and rightmost are terminal, with -1 and 1 reward respectively)
    Launches a Policy Iteration Algorithm in order to find the Optimal Policy and its Value Function
    Returns the Policy (Pi(s,a)) and its Value Function (V(s))
    """

    # Initialisation de V(s)
    V = np.random.random((len(lwe.S),))
    gamma = 0.999
    # Exception des V terminaux
    V[0] = 0.0
    V[6] = 0.0

    # Evaluation de la stratégie
    while True:
        delta = 0
        for s in lwe.S:
            v = V[s]
            # V[s] = 0.0
            for a in lwe.A:
                total = 0.0
                for sp in lwe.S:
                    for r in range(len(lwe.R)):
                        # total += lwe.p[sp, a, s, r] * (lwe.R[r] + gamma * V[sp])
                        # print(
                            # f"lwe.R[r] ({type(lwe.R[r])}): {lwe.R[r]}",
                            # f"gamma ({type(gamma)}): {gamma}",
                            # f"V[sp] ({type(V[sp])}): {V[sp]}",
                            # sep="\n"
                        # )
                        # print("###############################")
                        # print(
                            # f"sp ({type(sp)}): {sp}",
                            # f"pi ({type(pi)}): {pi}",
                            # f"pi[s] ({type(pi[s])}): {pi[s]}",
                            # f"pi[s, a] ({type(pi[s, a])}): {pi[s, a]}",
                            # f"s ({type(s)}): {s}",
                            # f"s ({type(r)}): {r}",
                            # sep="\n"
                        # )

                        # print(f"p(s',r|s,a) ({type(lwe.p[sp, int(pi[s, a]), s, r])}): {lwe.p[sp, int(pi[s, a]), s, r]}")

                        total += lwe.p[sp, int(pi[s, a]), s, r] * (lwe.R[r] + gamma * V[sp])
                        # print("total: ",total)
                        pass
                # total *= pi[s, a]
                V[s] = total
                # print("V[s]: ", V[s])
            delta = max(delta, abs(v - V[s]))
        if delta < theta:
            break

    # Amélioration de la stratégie
    policy_stable = True
    for s in lwe.S:
        # old_action = pi[s]
        old_action = V[s]

        ###################

        for a in lwe.A:
            total = 0.0
            for sp in lwe.S:
                for r in range(len(lwe.R)):
                    total += lwe.p[sp, a, s, r] * (lwe.R[r] + gamma * V[sp])
                    pass
                #total *= pi[s, a]

                new_action = np.argmax(total)

            ###################
            """
            #for a in lwe.A:
            for sp in lwe.S:
                for r in range(len(lwe.R)):
                    total += lwe.p[sp, V, s, r] * (lwe.R[r] + gamma * V[sp])
            new_action = np.argmax(total)
            """
            # print("old_action : ", old_action, end="\n")
            # print("new_action : ", new_action)
            if old_action != new_action:
                policy_stable = False

        if policy_stable:
            break

    return V

def value_iteration_on_line_world():
    """
    Creates a Line World of 7 cells (leftmost and rightmost are terminal, with -1 and 1 reward respectively)
    Launches a Value Iteration Algorithm in order to find the Optimal Policy and its Value Function
    Returns the Policy (Pi(s,a)) and its Value Function (V(s))
    """
    lwe = LineWorldEnv()
    V = {s: 0 for s in lwe.S}

    while True:
        old_V = V.copy()

        for s in lwe.S:
            Q = {}
            for a in lwe.A:
                Q[a] = lwe.p * (lwe.R * old_V[s+1])  # self.p = np.zeros((len(self.S), len(self.A), len(self.S), len(self.R)))
                print("######")
                print(Q[a])
                print("######")

            V[s] = max(Q.values())

        if all(old_V[s] == V[s] for s in lwe.S):
            break

        return V

def policy_evaluation_on_line_world(pi: np.ndarray, lwe: LineWorldEnv, theta=0.0000001):
    """
        Creates a Line World of 7 cells (leftmost and rightmost are terminal, with -1 and 1 reward respectively)
        Launches a Policy Evaluation Algorithm in order to find the Value Function of a uniform random policy
        Returns the Value function (V(s)) of this policy
    """
    """
    _input:
        π: Stratégie (policy)

    _params: theta > 0

    """

    # Initialisation de V(s)
    V = np.random.random((len(lwe.S),))
    gamma = 0.999
    # Exception des V terminaux
    V[0] = 0.0
    V[6] = 0.0

    while True:
        delta = 0
        for s in lwe.S:
            v = V[s]
            V[s] = 0.0
            for a in lwe.A:
                total = 0.0
                for sp in lwe.S:
                    for r in range(len(lwe.R)):
                        total += lwe.p[sp, a, s, r] * (lwe.R[r] + gamma * V[sp])
                        pass
                total *= pi[s, a]
                V[s] += total
            delta = max(delta, abs(v - V[s]))
        if delta < theta:
            break

    return V

def action_values(lwe: LineWorldEnv, state, V, gamma=1):
    B = np.zeros(len(lwe.A))
    for i in lwe.A:  # 0, 1
        print(f"==> is is: {i}")
        for probability, next_state, reward in lwe.p[state][i]:
            print(f"==> je suis la {lwe.p[state][i]}")
            B[i] += probability * (reward + gamma * V[next_state])
    return B

def policy_improvement(lwe: LineWorldEnv, V):
    policy = np.ones((len(lwe.S), len(lwe.A))) * 1 / (len(lwe.S) - len(lwe.A))
    policy_stable = True

    for s in lwe.S:
        pi_dict = dict()
        old_action = []
        for a in lwe.A:
            # Find the best action by one-step lookahead
            old_action.append(action_values(lwe, s, V))  # policy

        # take the best action for the current policy
        best_action = np.argmax(old_action)

        # Update policy
        # with if, and upgrade policy

        # if policy stable shutdown

        # pi_dict[s] = {0: old_action, 1: best_action}

        if old_action != best_action:
            policy_stable = False
        policy[s] = np.sum([np.eye(lwe.A)[i] for i in best_action], axis=0) / len(best_action)
    return policy

#####################################
#####################################
#####################################

lwe = LineWorldEnv()
pi = np.ones((len(lwe.S), len(lwe.A))) * 0.5
V = np.zeros((len(lwe.S),))
gamma = 0.99 #facteur de remise du return
theta = 0.00001 #seuil de similitude requis pour stopper les updates

action_value = policy_evaluation(lwe, V, gamma, theta)
print("pi: ", pi)
print("V: ", V)
print("action_value: ", action_value)

policy_eval = policy_evaluation(lwe, V, gamma, theta)
print("policy_eval: ", pi)

# List of calls to debug in class functions
"""
    right_pi = np.zeros((len(LineWorldEnv().S), len(LineWorldEnv().A)))
    right_pi[:, 1] = 1.0

    left_pi = np.zeros((len(LineWorldEnv().S), len(LineWorldEnv().A)))
    left_pi[:, 0] = 1.0

    random_pi = np.random.random((len(LineWorldEnv().S), len(LineWorldEnv().A))) #* 0.5

    #random_pi = np.zeros((len(LineWorldEnv().S), len(LineWorldEnv().A)))  # * 0.5
    random_pi = np.ones((len(LineWorldEnv().S), len(LineWorldEnv().A)))  # * 0.5

    random_pi = np.random.random((len(LineWorldEnv().S), len(LineWorldEnv().A)))

    random_pi = np.random.random_integers(0, high=1, size=(len(LineWorldEnv().S), len(LineWorldEnv().A)))

    # print(f"Stratégie tout le temps aller à droite: ", policy_evaluation_on_line_world(pi=right_pi, lwe=LineWorldEnv()))
    # print(f"Stratégie tout le temps aller à gauche: ", policy_evaluation_on_line_world(pi=left_pi, lwe=LineWorldEnv()))
    # print(f"Stratégie aléatoire: ", policy_evaluation_on_line_world(pi=random_pi, lwe=LineWorldEnv()))

    # test_pi = np.ones((len(LineWorldEnv().S), len(LineWorldEnv().A)))
    # print(f"Stratégie aléatoire: ", policy_iteration_on_line_world(pi=test_pi, lwe=LineWorldEnv()))


    #print(policy_evaluation_on_line_world())

    #V = policy_evaluation_on_line_world()

    #lwe = LineWorldEnv()
    #Q = np.zeros([len(lwe.S), len(lwe.A)])
    #for i in range(len(lwe.S)):
    #    Q[i] = action_values(lwe,i, V)
    #print("Action-Value Function:")
    #print(Q)

    print(policy_iteration_on_line_world(pi=random_pi, lwe=LineWorldEnv()))
"""