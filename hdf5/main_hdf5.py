# import sys
# sys.path.insert(1, "/Users/ischoning/PycharmProjects/eyeMovements/eventdetect-master")
import pandas as pd
from pylab import *
import matplotlib.pyplot as plt
import numpy as np
from hdf5 import plots_hdf5, preprocessing_hdf5


def clean_sequence(df):
    # determine the sequence of events
    prev_st = df.hidden_state[0]
    sequence = []
    count = 1
    for i in range(1, len(df)):
        st = df.hidden_state[i]
        if st == prev_st:
            count += 1
        elif st != prev_st:
            sequence.append([prev_st, count])
            count = 1
            prev_st = st
        if i == len(df) - 1:
            sequence.append([prev_st, count])

    print("Sequence:", sequence)

    # remove sequence if it contains fewer than 4 data points
    # Note: not possible to have two contiguous saccades without a fixation (or smooth pursuit)
    # in between
    s = np.array(sequence)
    s = s[:,1].astype(int)
    indices = np.where(s < 4)
    for i in indices[0]:
        start = np.sum(s[:i])
        end = np.sum(s[:i+1])
        df.drop(df.index[start:end], axis = 0, inplace = True)
        del sequence[i]
    df.reset_index(drop=True, inplace=True)

    print("Sequence after removing events less than 4 samples long:", sequence)

    return df


def label_event(event, states):
    if len(states) == 0:
        print("No states provided.")
        return
    if event == 1:
        return states[1] # 'Fixation'
    if event == 0:
        return states[0] # 'Saccade'
    return 'Other'


def print_events(t, event, states):
    events = []
    event_label = label_event(event.iloc[0], states)
    start = t.iloc[0]
    n = 1
    event_change = False
    for i in range(len(event)):
        if i == len(event)-1 or event.iloc[i] != event.iloc[i+1]:
            event_change = True
        if event_change:
            end = t.iloc[i]
            events.append([event_label, n, start, end])
            if i != len(event)-1:
                start = t.iloc[i+1]
                event_label = label_event(event.iloc[i+1], states)
                n = 1
                event_change = False
        else:
            n += 1

    print('=====================================================================')

    for e in events:
        print("%s for %d samples from %d ms to %d ms." % (e[0], e[1], e[2], e[3]))

    print('=====================================================================')

    print('Number of Fixation events:', sum(event==1))
    print('Number of Saccade events:', sum(event==0))
    print('Total number of events:', len(event))

    return None


def gaussian(mu, sigma, x):
    return 1.0/sqrt(2*pi*sigma**2)*exp(-(x-mu)**2/(2*sigma**2))


def sequence(df):
    # determine the sequence of events
    prev_st = df.event[0]
    prev_t = df.time[0]
    prev_amp = df.d[0]
    sequence = []
    count = 1
    for i in range(1, len(df)):
        st = df.event[i]
        if st == prev_st:
            count += 1
        elif st != prev_st:
            t = (df.time[i-1] - prev_t)*1000
            amp = abs(prev_amp - df.d[i-1])
            sequence.append({'State':prev_st, 'Num_samples':count, 'Amplitude':amp, 'Duration_ms':t})
            count = 1
            prev_st = st
            prev_t = df.time[i]
            prev_amp = df.d[i]
        if i == len(df) - 1:
            t = (df.time[i-1] - prev_t) * 1000
            amp = abs(prev_amp - df.d[i-1])
            sequence.append({'State':prev_st, 'Num_samples':count, 'Amplitude':amp, 'Duration_ms':t})

    print("Sequence:", sequence)

    return sequence


def main():

    #### STEP 0: Load Data ####
    # files
    file = '/Users/ischoning/PycharmProjects/GitHub/data/hdf5/events_58/data_collection_events_eyetracker_MonocularEyeSampleEvent.csv'

    # create dataframe
    df = pd.read_csv(file, sep=";", float_precision=None)


    #### STEP 1: Clean Outliers ####
    df = preprocessing_hdf5.remove_outliers(df, process = True)

    # plot results after removing outliers
    plots_hdf5.plot_path(df)
    plots_hdf5.plot_vs_time(df, feat = df.d, label ='Amplitude')
    plots_hdf5.plot_vs_time(df, feat = df.vel, label ='Velocity')


    #### STEP 2: Filter Fixations ####

    # plot velocity histogram
    hist, bin_edges = plots_hdf5.plot_hist(df, title='Velocity', x_axis='deg/s')

    # sanity check:
    prob = hist/len(hist)
    print(np.sum(prob))

    # calculate distribution characteristics
    mu = np.mean(df.vel)
    sigma = np.std(df.vel)
    med = np.median(df.vel)
    print("mean:", mu, "std:", sigma)
    print("median:", med)

    # create fixation gaussian
    fix = df.vel[df.vel<=mu]
    mu_fix = np.mean(fix)
    sigma_fix = np.std(fix)
    med_fix = np.median(fix)
    print("fix mean:", mu_fix, "fix std:", sigma_fix)
    print("fix median:", med_fix)
    plt.hist(fix, bins = int(len(fix)/2), normed=True)
    y = gaussian(mu_fix, sigma_fix, fix)
    plt.plot(fix, y, color = 'green')
    plt.title('Fixations')
    plt.show()

    # basic threshold classification
    df['event'] = np.where(df.vel <= mu_fix+3*sigma_fix, 'Fix', 'Sac')

    # plot classification
    plots_hdf5.plot_events(df)

    #### FILTER SACCADES ####

    # create non-fixation gaussian
    sac = df.vel[df.vel > mu]
    mu_sac = np.mean(sac)
    sigma_sac = np.std(sac)
    med_sac = np.median(sac)
    print("non-fix mean:", mu_sac, "non-fix std:", sigma_sac)
    print("non-fix median:", med_sac)
    plt.hist(sac, bins=int(len(sac) / 2), normed=True)
    y = gaussian(mu_sac, sigma_sac, sac)
    plt.plot(sac, y, color='green')
    plt.title('Non-Fixations')
    plt.show()

    # basic threshold classification
    df['event2'] = np.where(df.vel > mu_sac + 3 * sigma_sac, 'Sac', 'SmP')
    df.event = np.where(df.event != 'Fix', df.event2, 'Fix')

    # plot classification
    plots_hdf5.plot_events(df)


    #### STEP 3: Filter Saccades Using Carpenter's Theorem ####

    # create sequence of events
    seq = pd.DataFrame(sequence(df))

    # plot amplitude and velocity of "Sac's" along with ideal (Carpenter's: D = 21 + 2.2A, D~ms, A~deg)
    sacs = seq[seq.State == 'Sac']
    plt.scatter(sacs.Amplitude, sacs.Duration_ms, label = "'non-fixation' samples")
    x = linspace(min(sacs.Amplitude),max(sacs.Amplitude))
    y = 21 + 2.2*x     # Carpenter's Theorem
    plt.plot(x, y, color = 'green', label = 'D = 21 + 2.2A')
    head = 'Saccades: Amplitude vs Duration'
    plt.title(head)
    plt.xlabel('amplitude (deg)')
    plt.ylabel('duration (ms)')
    plt.legend()
    plt.show()

    print("========= FULL SEQUENCE =========")
    print(seq)
    print("========= SAC SEQUENCE =========")
    print(sacs)

    # calculate error rate
    df['error'] = (abs(sacs.Duration_ms - (21+2.2*sacs.Amplitude))/(21+2.2*sacs.Amplitude))*100
    print("Average Error:", np.mean(df.error), "%")

    # classify "Sac" into "Sac" or "Other" depending on error rate with Carpenter's Theorem

"""

    # if velocity is greater than 3 standard deviations from the mean of the pmf, classify the point as saccade, else fixation
    # NOTE that the white space in the plot is due to jump in ms between events
    states = ['Sac', 'Fix']
    df['event'] = np.where(df.vel_l <= 0.02, 'Fix', 'Sac')

    print('=============== STEP 1: Filter Saccades ===============')

    # estimate priors (sample means)
    mean_fix = np.mean(df[df.event == 'Fix']['vel_r'])
    mean_sac = np.mean(df[df.event == 'Sac']['vel_r'])
    std_fix = np.std(df[df.event == 'Fix']['vel_r'])
    std_sac = np.std(df[df.event == 'Sac']['vel_r'])
    print("Fixation: mean =", mean_fix, "standard deviation =", std_fix)
    print("Saccade: mean =", mean_sac, "standard deviation =", std_sac)

    print('\n============== BEGIN VITERBI ==============')
    # first run EM to get best match params (priors, trans, emission probabilities)
    # then run Viterbi HMM algorithm to output the most likely sequence given the params calculated in EM
    obs = df.vel_l.astype(str)
    obs = obs.tolist()
    states = ['Sac', 'Fix']
    start_p = [0.5, 0.5]
    trans_p = np.array([[0.5, 0.5],
                        [0.5, 0.5]])
    Sac = []
    Fix = []
    for o in obs:
        x = float(o)
        if o not in Sac:
            Sac.append(gaussian(mean_sac, std_sac, x))
        if o not in Fix:
            Fix.append(gaussian(mean_fix, std_fix, x))

    emit_p = np.array([Sac, Fix])
    df['hidden_state'] = viterbi.run(obs, states, start_p, trans_p, emit_p)
    print(df['hidden_state'].value_counts())
    print('=============== END VITERBI ===============')

    print('\n============== BEGIN BAUM-WELCH ==============')
    trans_p, emit_p = baum_welch.run(obs, states, start_p, trans_p, emit_p)
    print('============== END BAUM-WELCH ==============')

    print('\n============== BEGIN UPDATED VITERBI ==============')
    df['hidden_state'] = viterbi.run(obs, states, start_p, trans_p, emit_p)
    print('=============== END UPDATED VITERBI ===============')

    df = clean_sequence(df)

    print('\n=============== STEP 2: Classify Fixations and Smooth Pursuits ===============')

    # filter out Saccades
    df = df[df.hidden_state != 'Sac']
    df.reset_index(drop=True, inplace=True)

    ang_vel = df['ang_vel']
    states = ['Smooth Pursuit', 'Fixation']
    plt.plot(df.time, df.vel_l)
    plt.show()

    df['fix1 smp0'] = np.where(df.vel_l <= 0.02, 1, 0)
    event = df['fix1 smp0']
    print_events(df.time, event=event, states=states)

    # estimate priors (sample means)
    mean_fix = np.mean(df[event == 'Fix']['vel_r'])
    mean_smp = np.mean(df[event == 'SmP']['vel_r'])
    std_fix = np.std(df[event == 'Fix']['vel_r'])
    std_smp = np.std(df[event == 'SmP']['vel_r'])
    print("Fixation: mean =", mean_fix, "standard deviation =", std_fix)
    print("Smooth Pursuit: mean =", mean_smp, "standard deviation =", std_smp)

    print('\n============== BEGIN VITERBI ==============')

    # first run EM to get best match params (priors, trans, emission probabilities)
    # then run Viterbi HMM algorithm to output the most likely sequence given the params calculated in EM
    obs = ang_vel.astype(str)
    obs = obs.tolist()
    states = ['SmP', 'Fix']
    # p = math.log(0.5)
    start_p = [0.5, 0.5]
    trans_p = np.array([[0.5, 0.5],
                        [0.5, 0.5]])
    # Note: not possible to have two contiguous saccades without a fixation (or smooth pursuit)
    # in between
    SmP = []
    Fix = []
    for o in obs:
        x = float(o)
        if o not in SmP:
            SmP.append(gaussian(mean_sac, std_sac, x))
        if o not in Fix:
            Fix.append(gaussian(mean_fix, std_fix, x))

    emit_p = np.array([SmP, Fix])
    df['hidden_state'] = viterbi.run(obs, states, start_p, trans_p, emit_p)
    print(df['hidden_state'].value_counts())

    print('=============== END VITERBI ===============')

    print('\n============== BEGIN BAUM-WELCH ==============')

    trans_p, emit_p = baum_welch.run(obs, states, start_p, trans_p, emit_p)

    print('============== END BAUM-WELCH ==============')

    print('\n============== BEGIN UPDATED VITERBI ==============')

    df['hidden_state'] = viterbi.run(obs, states, start_p, trans_p, emit_p)
    # print(len(df['hidden_state']))
    print(df['hidden_state'].value_counts())

    print('=============== END UPDATED VITERBI ===============')

"""

if __name__ == "__main__":
    # Testing
    # hello("Isabella")
    main()
