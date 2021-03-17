# import sys
# sys.path.insert(1, "/Users/ischoning/PycharmProjects/eyeMovements/eventdetect-master")
import pandas as pd
from pylab import *
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import math
import matplotlib.transforms as mtransforms
import hmm
import baum_welch
import settings


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
    print("hist sum:", np.sum(hist * np.diff(bin_edges))) # sanity check - should = 1

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

def gaussian(mu, sigma, x):
    return 1/sqrt(2*pi*sigma**2)*exp(-(x-mu)**2/(2*sigma**2))

def normalize(d):
    '''
    input: dictionary
    returns: normalized dictionary (sum of values = 1)
    '''
    #total = sum(d.values()) # sum not working...
    dictlist = []
    for value in d.values():
        dictlist.append(value)
    total = sum(dictlist)
    d = {k: v / total for k, v in d.items()}
    return d

def main():

    # files
    df = pd.read_csv('/Users/ischoning/PycharmProjects/GitHub/data/participant08_preprocessed172.csv')

    # shorten dataset for better visuals and quicker results
    #df = df[int(len(df)/200):int(len(df)/100)]
    df = df[100:int(len(df)/500)]

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

    # show histogram of angular velocity
#    make_hist(ang_vel, 'Histogram of Angular Velocity', 'angular velocity', 'number of occurrences')

    # make pmf
    pmf(ang_vel, 'PMF of Angular Velocity', 'angular velocity', 'probability')

    # if velocity is greater than 3 standard deviations from the mean of the pmf, classify the point as saccade, else fixation
    # NOTE that the white space in the plot is due to jump in ms between events
    states = ['Saccade', 'Fixation']
    df['fix1 sac0'] = np.where(ang_vel <= 0.02, 1, 0)
    event = df['fix1 sac0']
    plot_vs_time(t, ang_vel, y=[], title='Combined Angular Velocity Over Time', y_axis='degrees per millisecond', event = event)
    print_events(t, event = event, states = states)

    # estimate priors (sample means)
    mean_fix = np.mean(df[event==1]['ang_vel'])
    mean_sac = np.mean(df[event==0]['ang_vel'])
    std_fix = np.std(df[event==1]['ang_vel'])
    std_sac = np.std(df[event==0]['ang_vel'])
    print("Fixation: mean =", mean_fix, "standard deviation =", std_fix)
    print("Saccade: mean =", mean_sac, "standard deviation =", std_sac)

    ang_vel.to_pickle('ang_vel.pkl')

    # assuming underlying distrib is gaussian, find MLE mu and sigma

    print('============== BEGIN VITERBI ==============')

    # first run EM to get best match params (priors, trans, emission probabilities)
    # then run Viterbi HMM algorithm to output the most likely sequence given the params calculated in EM
    obs = ang_vel.astype(str)
    obs = obs.tolist()
    states = ['Sac', 'Fix']
    start_p = {"Sac": 0.5, "Fix": 0.5}
    trans_p = {
        "Sac": {"Sac": 0.5, "Fix": 0.5},
        "Fix": {"Sac": 0.5, "Fix": 0.5},
    }
    Sac = {}
    Fix = {}
    for o in obs:
        x = float(o)
        if o not in Sac:
            Sac[o] = gaussian(mean_sac, std_sac, x)
        if o not in Fix:
            Fix[o] = gaussian(mean_fix, std_fix, x)
    # normalize
    Sac = normalize(Sac)
    Fix = normalize(Fix)
    emit_p = {"Sac": Sac, "Fix": Fix}
    df['hidden_state'] = hmm.viterbi(obs, states, start_p, trans_p, emit_p)
    #print(len(df['hidden_state']))

    print('=============== END VITERBI ===============')

    # Q's: Why is probability so small? Should I be working with logs?

    settings.init()

    print('============== BEGIN BAUM-WELCH ==============')

    baum_welch.run(obs, states, start_p, trans_p, emit_p)

    print('============== END BAUM-WELCH ==============')


if __name__ == "__main__":
    # Testing
    # hello("Isabella")
    main()
