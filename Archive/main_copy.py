# import sys
# sys.path.insert(1, "/Users/ischoning/PycharmProjects/eyeMovements/eventdetect-master")
import pandas as pd
from pylab import *
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import math
import matplotlib.transforms as mtransforms
import viterbi
import baum_welch


def show_path(Ax, Ay):
    plt.scatter(Ax, Ay)
    plt.title('Angular Visual Movement (Degrees)')
    plt.xlabel('degrees')
    plt.ylabel('degrees')
    plt.show()

    return None

def plot_eye_path(df):
    '''
    input: pandas dataframe that includes 'Left(Right)EyeForward_x(y)' in header
    output: two scatter plots (x,y movements of each eye)
    return: None
    '''
    fig, ax = plt.subplots(1,2, sharex = True, sharey = True)

    ax[0].scatter(x=df['Ax_left'], y=df['Ay_left'])
    ax[0].set_title('Left Eye Displacement')

    ax[1].scatter(x=df['Ax_right'], y=df['Ay_right'])
    ax[1].set_title('Right Eye Displacement')

    plt.xlabel('degrees')
    plt.ylabel('degrees')

    plt.show()

    return None

def plot_vs_time(t, x, y = [], title = None, y_axis = None, event = None):
    '''
    input: two values (x,y) to be plotted against time
    return: None
    '''

    fig, ax = plt.subplots()

    if len(y) == 0:
        ax.plot(t, x, color = 'red')

        # https://matplotlib.org/3.1.1/gallery/lines_bars_and_markers/fill_between_demo.html
        if event is not None:
            trans = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
            min_val = min(x)
            max_val = max(x)
            ax.fill_between(t, min_val, max_val, where = event == 1,
                            facecolor='green', alpha=0.5, transform=trans, label='fixation')
            ax.fill_between(t, min_val, max_val, where = event == 0,
                            facecolor='red', alpha=0.5, transform=trans, label='saccade')
            ax.legend()
    else:
        ax.plot(t, x, color = 'r', label = 'x')
        ax.plot(t, y, color = 'b', label = 'y')
        ax.legend()

    ax.set_title(title)
    ax.set_xlabel('timestamp (ms)')
    ax.set_ylabel(y_axis)

    plt.show()

    return None


def make_hist(data, title, x_axis, y_axis, density = False):
    if density:
        result, bin_edges = np.histogram(data, bins=len(data), density = True)
        plt.hist(data, bins=len(data), density = True)
    else:
        plt.hist(data, bins=int(len(data)/2))
    plt.title(title)
    plt.ylabel(y_axis)
    plt.xlabel(x_axis)
    plt.show()

    if density:
        return result, bin_edges
    else: return None

def pmf(data, title, x_axis, y_axis):
    hist, bin_edges = make_hist(data, title, x_axis, y_axis, density = True)
    #print("hist sum:", np.sum(hist * np.diff(bin_edges))) # sanity check - should = 1

    return None


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

def dict_to_list(d):
    '''
    :param d: dictionary
    :return: list of values in dictionary
    '''
    dictlist = []
    for value in d.values():
        dictlist.append(value)
    return dictlist

def gaussian(mu, sigma, x):
    return 1.0/sqrt(2*pi*sigma**2)*exp(-(x-mu)**2/(2*sigma**2))

def normalize(d):
    '''
    input: dictionary
    returns: normalized dictionary (sum of values = 1)
    '''
    #total = sum(d.values()) # sum not working...
    dictlist = dict_to_list(d)
    total = sum(dictlist)
    d = {k: v / total for k, v in d.items()}
    return d

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


def run_dispersion(df, eye, ws = 0, thresh = 0, method='IDT'):
    # inspired by Algorithm 1 in A. George, A. Routray and applied to I-DT
    # 29 may 2021

    mdf = 0.04 # minimum duration fixation threshold (40 ms - Houpt)
    lastState = 'other'
    fixStartTime = df.loc[0,'time']
    fixStartLoc = 0
    N = len(df)
    res = ['other']
    for i in range(1,N):
        disp = df.loc[i,'d']
        if disp < thresh:
            currentState = 'fix'
            if lastState != currentState:
                fixStartLoc = i
                fixStartTime = df.loc[i,'time']
        else:
            if lastState == 'fix':
                duration = df.loc[i,'time']-fixStartTime
                if duration < mdf:
                    for j in range(fixStartLoc,i):
                        res.append('other')
            currentState = 'other'
        lastState = currentState
        res.append(currentState)

    return res


def main():

    # files
    df = pd.read_csv('/Users/ischoning/PycharmProjects/GitHub/data/participant08_preprocessed172.csv')

    # shorten dataset for better visuals and quicker results
    #df = df[int(len(df)/200):int(len(df)/100)]
    df = df[100:int(len(df)/500)]
    df.reset_index(drop=True, inplace=True)

    # assign relevant data
    lx = df['left_forward_x']
    ly = df['left_forward_y']
    lz = df['left_forward_z']
    rx = df['right_forward_x']
    ry = df['right_forward_y']
    rz = df['right_forward_z']
    t = df['timestamp_milis']

    # compute angular values
    df['Ax_left'] = np.rad2deg(np.arctan2(lx, lz))
    df['Ay_left'] = np.rad2deg(np.arctan2(ly, lz))
    df['Ax_right'] = np.rad2deg(np.arctan2(rx, rz))
    df['Ay_right'] = np.rad2deg(np.arctan2(ry, rz))
    df['Avg_angular_x'] = df[['Ax_left', 'Ax_right']].mean(axis=1)
    df['Avg_angular_y'] = df[['Ay_left', 'Ay_right']].mean(axis=1)

    # show vision path in averaged angular degrees
#    show_path(df['Avg_angular_x'], df['Avg_angular_y'])
    # print('Length of capture time:', len(t))
    # print('Length of capture time differences:',
    #       len(np.diff(t/1000000)))

    # # show vision path, separately for each eye
#    plot_eye_path(df)

    # show angular displacement over time, averaged over both eyes
#    plot_vs_time(t, df['Avg_angular_x'], df['Avg_angular_y'], 'Angular Displacement Over Time', 'degrees')

    # plot angular velocity for x and y
    # remove the last row so lengths of each column are consistent
    dt = np.diff(t)  # aka isi
    dx = np.diff(df['Avg_angular_x'])
    dy = np.diff(df['Avg_angular_y'])

    df.drop(df.tail(1).index, inplace=True)
    t = df['timestamp_milis']

#    plot_vs_time(t,dx/dt,dy/dt, 'Angular Velocity Over Time', 'degrees per millisecond')

    # plot combined angular velocity
    df['ang_vel'] = np.sqrt(np.square(dx) + np.square(dy))
    ang_vel = df['ang_vel']
#    plot_vs_time(t, ang_vel, y = [], title = 'Combined Angular Velocity Over Time', y_axis = 'degrees per millisecond')

    plt.scatter(range(len(ang_vel)), ang_vel)
#    plt.show()
    plt.scatter(ang_vel, ang_vel)
#    plt.show()

    # plot angular acceleration for x and y
    # remove the last row so lengths of each column are consistent
    dt = np.diff(t)  # aka isi
    dv = np.diff(df['ang_vel'])

    df_ = df.copy()
    df_.drop(df_.tail(1).index, inplace=True)
    df_['ang_acc'] = dv/dt

    # plot combined angular accleration
#    plot_vs_time(df_['timestamp_milis'],df_['ang_vel'], y = df_['ang_acc'], title = 'Combined Angular Acceleration Over Time', y_axis = 'degrees per millisecond')

    # show histogram of angular velocity
    make_hist(ang_vel, 'Histogram of Angular Velocity', 'angular velocity', 'number of occurrences')

    # make pmf
    pmf(ang_vel, 'PMF of Angular Velocity', 'angular velocity', 'probability')

    # if velocity is greater than 3 standard deviations from the mean of the pmf, classify the point as saccade, else fixation
    # NOTE that the white space in the plot is due to jump in ms between events
    states = ['Saccade', 'Fixation']
    df['fix1 sac0'] = np.where(ang_vel <= 0.02, 1, 0)
    event = df['fix1 sac0']
#    plot_vs_time(t, ang_vel, y=[], title='Combined Angular Velocity Over Time', y_axis='degrees per millisecond', event = event)
    print_events(t, event = event, states = states)

    print('=============== STEP 1: Filter Saccades ===============')

    # estimate priors (sample means)
    mean_fix = np.mean(df[event == 1]['ang_vel'])
    mean_sac = np.mean(df[event == 0]['ang_vel'])
    std_fix = np.std(df[event == 1]['ang_vel'])
    std_sac = np.std(df[event == 0]['ang_vel'])
    print("Fixation: mean =", mean_fix, "standard deviation =", std_fix)
    print("Saccade: mean =", mean_sac, "standard deviation =", std_sac)

    print('\n============== BEGIN VITERBI ==============')
    # first run EM to get best match params (priors, trans, emission probabilities)
    # then run Viterbi HMM algorithm to output the most likely sequence given the params calculated in EM
    obs = ang_vel.astype(str)
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
    plt.plot(df.timestamp_milis, ang_vel)
    plt.show()

    df['fix1 smp0'] = np.where(ang_vel <= 0.02, 1, 0)
    event = df['fix1 smp0']
    print_events(t, event=event, states=states)

    # estimate priors (sample means)
    mean_fix = np.mean(df[event == 1]['ang_vel'])
    mean_smp = np.mean(df[event == 0]['ang_vel'])
    std_fix = np.std(df[event == 1]['ang_vel'])
    std_smp = np.std(df[event == 0]['ang_vel'])
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

    df = clean_sequence(df)


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
