'''
source: https://en.wikipedia.org/wiki/Hidden_Markov_model
Hidden Markov Model (HMM) is a statistical Markov model in which the system being modeled is assumed to be a Markov
process (X) with unobservable 'hidden' states.
(A Markov chain is a stochastic model describing a sequence of possible events in which the probability of each event
depends only on the state attained in the previous event.)
HMM assumes that there is another process Y whose behavior 'depends' on X.
The goal is to learn about X (eye movement event) by observing Y (ang displacement).
HMM stipulates that, for each time instance n0, the conditional probability distribution of Y_n0 given the history
{X_n = x_n}_n<=n0 must not depend on {x_n}_n<n0 (must not depend on states previous to n-1, that directly previous).
The value of the observed variable y(t) only depends on the value of the hidden variable x(t) at time t.

Let X_n and Y_n be discrete-time stochastic processes and n>=1. The pair (X_n, Y_n) is a hidden markov model if:
    - X_n is a Markov process whose behavior is not directly observable ('hidden')
    - Prob(Y_n in A | X_1 = x_1,...,X_n=x_n) = Prob(Y_n in A | X_n = x_n) for every n>=1, x_1,...,x_n, and an
    arbitrary (measurable) set A.

Terminology:
> The states of the process X_n are called hidden states, and Prob(Y_n in A | X_n = x_n) is called emission probability
or output probability.
> Markov matrix is the N x N matrix of transition probabilities. (sums to 1)

* Our goal is the find the most likely sequence of events (fixations or saccades) by evaluating the joint probability
of both the state sequency and the observations (angular displacement) for each case.
ie We want to find the most likely explanation (maximum) over all possible state sequences.
For cases involving finding the most likely explanation for an observation sequence, we use the Viterbi algorithm.

Two states: fixation (F) or saccade (S)
Find: Prob(F|F), Prob(F|S), Prob(S|S), Prob(S|F)

--------------------- BEGIN EXAMPLE ------------------------
states = ('Rainy', 'Sunny')

observations = ('walk', 'shop', 'clean')

start_probability = {'Rainy': 0.6, 'Sunny': 0.4}

transition_probability = {
   'Rainy' : {'Rainy': 0.7, 'Sunny': 0.3}, # there is 30% chance that tomorrow will be sunny if today is rainy
   'Sunny' : {'Rainy': 0.4, 'Sunny': 0.6},
   }

emission_probability = {
   'Rainy' : {'walk': 0.1, 'shop': 0.4, 'clean': 0.5},
   'Sunny' : {'walk': 0.6, 'shop': 0.3, 'clean': 0.1},
   }
---------------------- END EXAMPLE -------------------------
'''

'''
VITERBI ALGORITHM:
A dynamic programming algorithm for finding the most likely sequence of hidden states (Viterbi path) that results in a
sequence of observed events, especially in the context of Markov information sources and hidden Markov models.

source: https://en.wikipedia.org/wiki/Viterbi_algorithm#Example
'''

def viterbi(obs, states, start_p, trans_p, emit_p):
    V = [{}]
    for st in states:
        V[0][st] = {"prob": start_p[st] * emit_p[st][obs[0]], "prev": None}
    # Run Viterbi when t > 0
    for t in range(1, len(obs)):
        V.append({})
        for st in states:
            max_tr_prob = V[t - 1][states[0]]["prob"] * trans_p[states[0]][st]
            prev_st_selected = states[0]
            for prev_st in states[1:]:
                tr_prob = V[t - 1][prev_st]["prob"] * trans_p[prev_st][st]
                if tr_prob > max_tr_prob:
                    max_tr_prob = tr_prob
                    prev_st_selected = prev_st

            max_prob = max_tr_prob * emit_p[st][obs[t]]
            V[t][st] = {"prob": max_prob, "prev": prev_st_selected}

    for line in dptable(V):
        print(line)

    opt = []
    max_prob = 0.0
    best_st = None
    # Get most probable state and its backtrack
    for st, data in V[-1].items():
        if data["prob"] > max_prob:
            max_prob = data["prob"]
            best_st = st
    opt.append(best_st)
    previous = best_st

    # Follow the backtrack till the first observation
    for t in range(len(V) - 2, -1, -1):
        opt.insert(0, V[t + 1][previous]["prev"])
        previous = V[t + 1][previous]["prev"]

    print ("The steps of states are " + " ".join(opt) + " with highest probability of %s" % max_prob)


def dptable(V):
    # Print a table of steps from dictionary
    yield " ".join(("%12d" % i) for i in range(len(V)))
    for state in V[0]:
        yield "%.7s: " % state + " ".join("%.7s" % ("%f" % v[state]["prob"]) for v in V)


obs = ("normal", "cold", "dizzy")
states = ("Healthy", "Fever")
start_p = {"Healthy": 0.6, "Fever": 0.4}
trans_p = {
    "Healthy": {"Healthy": 0.7, "Fever": 0.3},
    "Fever": {"Healthy": 0.4, "Fever": 0.6},
}
emit_p = {
    "Healthy": {"normal": 0.5, "cold": 0.4, "dizzy": 0.1},
    "Fever": {"normal": 0.1, "cold": 0.3, "dizzy": 0.6},
}

viterbi(obs,
        states,
        start_p,
        trans_p,
        emit_p)


