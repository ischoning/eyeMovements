import math

# https://en.wikipedia.org/wiki/Viterbi_algorithm#Example
def run(obs, states, start_p, trans_p, emit_p):
    V = [{}]

    i = 0
    for st in states:
        V[0][st] = {"prob": start_p[i] * emit_p[i][0], "prev": None}
        i += 1

    # Run Viterbi when t > 0
    for t in range(1, len(obs)):
        V.append({})

        for i in range(len(states)):
            max_tr_prob = V[t - 1][states[0]]["prob"] * trans_p[0][i]
            prev_st_selected = states[0]
            j = 0
            for prev_st in states[1:]:
                tr_prob = V[t - 1][prev_st]["prob"] * trans_p[j][i]
                if tr_prob > max_tr_prob:
                    max_tr_prob = tr_prob
                    prev_st_selected = prev_st
                j += 1

            max_prob = max_tr_prob * emit_p[i][t]
            V[t][states[i]] = {"prob": max_prob, "prev": prev_st_selected}

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

    print("The steps of states are " + " ".join(opt) + " with highest probability of %s" % max_prob)
    #print("The steps of states are " + " ".join(opt))

    return opt


def dptable(V):
    # Print a table of steps from dictionary
    yield " ".join(("%12d" % i) for i in range(len(V)))
    for state in V[0]:
        yield "%.7s: " % state + " ".join("%.7s" % ("%f" % v[state]["prob"]) for v in V)
