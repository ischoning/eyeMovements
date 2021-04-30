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
from scipy.interpolate import make_interp_spline, BSpline
import preprocessing


def show_path(x, y, targetx, targety):
    fig, ax = plt.subplots(1,1)
    ax.scatter(x, y, s = 0.5, label='participant')
    ax.scatter(targetx, targety, s = 10, c = 'orange', label = 'target')
    ax.set_title('Angular Visual Movement')
    ax.set_xlabel('degrees')
    ax.set_ylabel('degrees')
    ax.legend()
    fig.show()

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

def plot_vs_time(t, x, y = [], title = None, x_axis = None, y_axis = None, x_label = None, y_label = None, event = None):
    '''
    input: two values (x,y) to be plotted against time
    return: None
    '''
    if y_label is None:
        y_label = 'y'
    if x_label is None:
        x_label = 'x'
    if x_axis is None:
        x_axis = 'timestamp (ms)'

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
        ax.plot(t, x, color = 'r', label = x_label)
        ax.plot(t, y, color = 'b', label = y_label)
        ax.legend()

    ax.set_title(title)
    ax.set_xlabel(x_axis)
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

def overview_plot(data):
    fig, ax = plt.subplots()
    fig.subplots_adjust(right=0.75)

    ax1 = ax.twinx()
    ax2 = ax.twinx()

    # Offset the right spine of twin2.  The ticks and label have already been
    # placed on the right by twinx above.
    #ax2.spines.right.set_position(("axes", 1.2))

    p1, = ax.plot(data.sample_i, data.position, "b-", label="Displacement")
    p2, = ax1.plot(data.sample_i, data.velocity, "r-", label="Velocity")
    p3, = ax2.plot(data.sample_i, data.acceleration, "g-", label="Acceleration")

    # ax.set_xlim(0, 2)
    # ax.set_ylim(0, 2)
    # ax1.set_ylim(0, 4)
    # ax2.set_ylim(1, 65)

    ax.set_xlabel("sample sequential number")
    ax.set_ylabel("Degree")
    ax1.set_ylabel("Deg/Ms")
    ax2.set_ylabel("Deg/Ms^2")

    ax.yaxis.label.set_color(p1.get_color())
    ax1.yaxis.label.set_color(p2.get_color())
    ax2.yaxis.label.set_color(p3.get_color())

    tkw = dict(size=4, width=1.5)
    ax.tick_params(axis='y', colors=p1.get_color(), **tkw)
    ax1.tick_params(axis='y', colors=p2.get_color(), **tkw)
    ax2.tick_params(axis='y', colors=p3.get_color(), **tkw)
    ax.tick_params(axis='x', **tkw)

    ax.legend(handles=[p1, p2, p3])

    plt.show()


def main():
    preprocessing

    """
    # files
    file = '/Users/ischoning/PycharmProjects/GitHub/data/Nils9_sample_level_clean.csv'

    # create dataframe
    df = pd.read_csv(file)

    # filter to one subject (No eye condition, No scotoma filter)
    df.eye_condition.values.astype(str)
    df.scotoma_filter.values.astype(str)
    df = df.loc[df.eye_condition.values == "None"]
    df = df.loc[df.scotoma_filter.values == "None"]
    df = df.loc[df.subject_id.values == df.subject_id.values[0]]

    df.isi_raw = pd.to_numeric(df.isi_raw, errors = 'coerce')
    df.new_angle_right_x = pd.to_numeric(df.new_angle_right_x, errors = 'coerce')
    df.new_angle_right_y = pd.to_numeric(df.new_angle_right_y, errors = 'coerce')
    df.new_angle_left_x = pd.to_numeric(df.new_angle_left_x, errors='coerce')
    df.new_angle_left_y = pd.to_numeric(df.new_angle_left_y, errors='coerce')
    df.new_samplevelocity_right = pd.to_numeric(df.new_samplevelocity_right, errors = 'coerce')
    df.new_samplevelocity_left = pd.to_numeric(df.new_samplevelocity_left, errors = 'coerce')

    df.reset_index(drop=True, inplace=True)
    df.round(decimals=4) # for faster computation

    print(len(df))
    print(df.columns.values)

    # %% FORMATTING

    df = df[700:1150]

    # select which eye to use ('right' or 'left')
    # eye = 'right'

    # displacement
    # col = 'new_angle_' + eye
    # x = df[col + '_x']
    # y = df[col + '_y']
    # velocity
    # v = df['new_samplevelocity_' + eye]

    # average across both eyes
    # displacement
    x = df[['new_angle_right_x', 'new_angle_left_x']].mean(axis=1)
    y = df[['new_angle_right_y', 'new_angle_left_y']].mean(axis=1)
    # velocity
    v = df['new_samplevelocity_right']
    df['ang_vel'] = v
    # acceleration
    df['new_sampleaccel_right'] = np.diff(df['new_samplevelocity_right'], prepend=0) / df.isi_raw
    df['new_sampleaccel_left'] = np.diff(df['new_samplevelocity_left'], prepend=0) / df.isi_raw
    a = df[['new_sampleaccel_right', 'new_sampleaccel_left']].mean(axis=1)
    df['ang_acc'] = a
#    plot_vs_time(df.time, a, title = 'Acceleration', y_axis = 'deg/ms^2')

    # %% PLOTS AND ANALYSIS

    # show vision path in averaged angular degrees
    print(x[1:10])
#    show_path(x, y, df.target_angle_x, df.target_angle_y)

    #%% Determine eye dominance

    plt.figure(figsize=(18, 6))
    plt.plot(df.time, df.target_angle_x, color = 'black', alpha=0.5, label='x target')
    plt.plot(df.time, df.target_angle_y, color='black', alpha=0.5, label='y target')
    plt.plot(df.time, df.new_angle_right_x, color = 'green', alpha = 1, label = 'x right', linewidth = 2)
    plt.plot(df.time, df.new_angle_right_y, color = 'green', alpha = 1, label = 'y right', linewidth = 2)
    plt.plot(df.time, df.new_angle_left_x, color = 'orange', alpha=0.75, label='x left', linewidth = 2)
    plt.plot(df.time, df.new_angle_left_y, color='orange', alpha=0.75, label='y left', linewidth = 2)

    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), shadow = True, ncol=3)
    plt.title('Position Over Time')
    plt.xlabel('time (ms)')
    plt.ylabel('degree')

    plt.show()

    # %%

    # show angular displacement over time, averaged over both eyes
    d = sqrt(x**2 + y**2)
    #plot_vs_time(df.time, x, y=[], title='X Displacement Over Time', y_axis='degrees')
    #plot_vs_time(df.time, x, y, title='Angular Displacement Over Time', y_axis='degrees')
#    plot_vs_time(df.time, d, y=[], title='Combined Angular Displacement Over Time', y_axis='degrees')


    # show angular velocity over time, averaged over both eyes
#    plot_vs_time(df.time, v, y=[], title='Right Eye Angular Velocity Over Time', y_axis='degrees per millisecond')
    plt.scatter(df.time, v, s=5)
#    plt.show()

    #%%
    # show histogram of angular velocity
#    make_hist(v, 'Histogram of Angular Velocity', 'angular velocity', 'number of occurrences')

    # make pmf
    #pmf(v, 'PMF of Angular Velocity', 'angular velocity', 'probability')

    #%%
    # source/inspiration: Komogortsev "Hierarchical HMM for Eye Movement Classification"
    """
    """
    STEP 0: Pre-processing
    Compute position, velocity, acceleration feature sequences.
    Select appropriate features for classification.
    """
    """
    data = pd.DataFrame(columns=['sample_i','position','velocity','acceleration'])
    n = 100 # sample length
    i = 0
    while i*n <= len(df)-n:
        s = pd.DataFrame({
            'sample_i':[i],
            'position':[x[i*n:i*n+n].mean()],
            'velocity':[v[i*n:i*n+n].mean()],
            'acceleration':[a[i*n:i*n+n].mean()]
        })
        data = data.append(s)
        i += 1

    print(len(data))

    # show plots
    # define x as equally spaced values between the min and max of original x
    xnew = np.linspace(data.sample_i.min(), data.sample_i.max(), 200)
    y_ax = ["degrees", "deg/ms", "deg/ms^2"]
    for i in range(1, len(data.columns)):
        feature = data.columns[i]
        # define spline
        spl = make_interp_spline(data.sample_i, data.loc[:,feature], k=3)
        y_smooth = spl(xnew)
 #       plot_vs_time(xnew, y_smooth, title = feature, x_axis="sample sequential number", y_axis=y_ax[i-1])
"""
    #%%
    """
    STEP 1: First HMM Classification
    Filter saccades.
    """


    #%%
    """
    STEP 2: Second HMM Classification
    Classify fixations and smooth pursuits.
    """

    #%%
    """
    STEP 3:  Merge
    Return list of complete classification results.
    """
"""
    # if velocity is greater than 3 standard deviations from the mean of the pmf, classify the point as saccade, else fi›xation
    # NOTE that the white space in the plot is due to jump in ms between events
    states = ['Saccade', 'Fixation', 'Smooth Pursuit']

# first run hierarchical HMM for between fixation and saccade
    df['sac0_fix1_sp2'] = np.where(v <= 0.02, 1, 0)
    event = df['sac0_fix1_sp2']

    # estimate priors (sample means)
    mean_sac = np.mean(df[event == 0]['ang_vel'])
    mean_fix = np.mean(df[event == 1]['ang_vel'])
    std_sac = np.std(df[event == 0]['ang_vel'])
    std_fix = np.std(df[event == 1]['ang_vel'])

# then filter between fixation and smooth pursuit
    df_original = df.copy()
    df.drop(df.loc[df.sac0_fix1_sp2.values == 0])
    df['sac0_fix1_sp2'] = np.where(v <= 0.02, 1, 0)

    event = df['sac0_fix1_sp2']
    plot_vs_time(df.time, v, y=[], title='Combined Angular Velocity Over Time', y_axis='degrees per millisecond',
                 event=event)
    print_events(df.time, event=event, states=states)

    # estimate priors (sample means)
    mean_sac = np.mean(df[event == 0]['ang_vel'])
    mean_fix = np.mean(df[event == 1]['ang_vel'])
    mean_sp = np.mean(df[event == 2]['ang_vel'])
    std_sac = np.std(df[event == 0]['ang_vel'])
    std_fix = np.std(df[event == 1]['ang_vel'])
    std_sp = np.std(df[event == 2]['ang_vel'])
    print("Fixation: mean =", mean_fix, "standard deviation =", std_fix)
    print("Saccade: mean =", mean_sac, "standard deviation =", std_sac)
    print("Smooth Pursuit: mean =", mean_sp, "standard deviation =", std_sp)

    print(len(event))
"""
"""
    # files
    Nils = '/Users/ischoning/PycharmProjects/GitHub/data/Nils9_sample_level_clean.csv'
    Marek = '/Users/ischoning/PycharmProjects/GitHub/data/participant08_preprocessed172.csv'

    file = Nils

    # create dataframe
    df = pd.read_csv(file)

    # depending on type of data
    if file == Nils:
        df.eye_condition.values.astype(str)
        df.scotoma_filter.values.astype(str)
        df = df.loc[df.eye_condition.values == "None"]
        df = df.loc[df.scotoma_filter.values == "None"]
        df = df.loc[df.subject_id.values == df.subject_id.values[0]]

        # select which eye to use ('right' or 'left')
        eye = 'right'

        # displacement
        col = 'new_angle_'+eye
        x = df[col+'_x']
        y = df[col+'_y']

        # veloctiy
        v = df['new_samplevelocity_'+eye]

    else:
        # shorten dataset for better visuals and quicker results
        df = df[100:int(len(df) / 500)]

        # assign relevant data
        lx = df['left_forward_x']
        ly = df['left_forward_y']
        lz = df['left_forward_z']
        rx = df['right_forward_x']
        ry = df['right_forward_y']
        rz = df['right_forward_z']
        df.rename(columns={'timestamp_milis': 'time'}, inplace=True)

        # compute angular values
        df['Ax_left'] = np.rad2deg(np.arctan2(lx, lz))
        df['Ay_left'] = np.rad2deg(np.arctan2(ly, lz))
        df['Ax_right'] = np.rad2deg(np.arctan2(rx, rz))
        df['Ay_right'] = np.rad2deg(np.arctan2(ry, rz))
        df['Avg_angular_x'] = df[['Ax_left', 'Ax_right']].mean(axis=1)
        df['Avg_angular_y'] = df[['Ay_left', 'Ay_right']].mean(axis=1)

        x = df['Avg_angular_x']
        y = df['Avg_angular_y']

        print(len(x), len(y))

        # calculate angular velocity
        # remove the last row so lengths of each column are consistent
        dx = np.diff(x)
        dy = np.diff(y)

        df.drop(df.tail(1).index, inplace=True)
        df['ang_vel'] = np.sqrt(np.square(dx) + np.square(dy))
        ang_vel = df['ang_vel']

        # find angular acceleration for x and y
        # remove the last row so lengths of each column are consistent
        dt = np.diff(df['time'])  # aka isi
        dv = np.diff(df['ang_vel'])

        df_ = df.copy()
        df_.drop(df_.tail(1).index, inplace=True)
        ang_acc = dv / dt

    # show vision path in averaged angular degrees
    show_path(x, y)
    # print('Length of capture time:', len(t))
    # print('Length of capture time differences:',
    #       len(np.diff(t/1000000)))

    # # show vision path, separately for each eye
›#    plot_eye_path(df)

    # show angular displacement over time, averaged over both eyes
    plot_vs_time(df.time, x, y, 'Angular Displacement Over Time', 'degrees')

    # plot angular velocity for x and y
    plot_vs_time(df.time,dx/dt,dy/dt, 'Angular Velocity Over Time', 'degrees per millisecond')

    # plot combined angular velocity
    plot_vs_time(df.time, ang_vel, y = [], title = 'Combined Angular Velocity Over Time', y_axis = 'degrees per millisecond')

    # plot combined angular accleration
    plot_vs_time(df.time[:-1],x = ang_vel[-1], y = ang_acc, title = 'Combined Angular Acceleration Over Time', y_axis = 'degrees per millisecond')

    # show histogram of angular velocity
    make_hist(ang_vel, 'Histogram of Angular Velocity', 'angular velocity', 'number of occurrences')

    # make pmf
    pmf(ang_vel, 'PMF of Angular Velocity', 'angular velocity', 'probability')

    # if velocity is greater than 3 standard deviations from the mean of the pmf, classify the point as saccade, else fixation
    # NOTE that the white space in the plot is due to jump in ms between events
    states = ['Saccade', 'Fixation']
    df['fix1 sac0'] = np.where(ang_vel <= 0.02, 1, 0)
    event = df['fix1 sac0']
    plot_vs_time(df.time, ang_vel, y=[], title='Combined Angular Velocity Over Time', y_axis='degrees per millisecond', event = event)
    print_events(df.time, event = event, states = states)

    # estimate priors (sample means)
    mean_fix = np.mean(df[event==1]['ang_vel'])
    mean_sac = np.mean(df[event==0]['ang_vel'])
    std_fix = np.std(df[event==1]['ang_vel'])
    std_sac = np.std(df[event==0]['ang_vel'])
    print("Fixation: mean =", mean_fix, "standard deviation =", std_fix)
    print("Saccade: mean =", mean_sac, "standard deviation =", std_sac)

    x = np.linspace(-3, 3, 0.1)
    params = {'Fix': [mean_fix, std_fix], 'Sac': [mean_sac, std_sac]}
    for param in params.values():
        plt.plot(x, gaussian(param[0], param[1], x))
#    plt.show()

    ang_vel.to_pickle('ang_vel.pkl')

    # assuming underlying distrib is gaussian, find MLE mu and sigma

    print('\n============== BEGIN VITERBI ==============')

    # first run EM to get best match params (priors, trans, emission probabilities)
    # then run Viterbi HMM algorithm to output the most likely sequence given the params calculated in EM
    obs = ang_vel.astype(str)
    obs = obs.tolist()
    states = ['Sac', 'Fix']
    #p = math.log(0.5)
    start_p = {"Sac": 0.5, "Fix": 0.5}
    trans_p = {
        "Sac": {"Sac": 0.5, "Fix": 0.5},
        "Fix": {"Sac": 0.5, "Fix": 0.5},
    }
    # Note: not possible to have two contiguous saccades without a fixation (or smooth pursuit)
    # in between
    Sac = {}
    Fix = {}
    for o in obs:
        x = float(o)
        if o not in Sac:
            Sac[o] = gaussian(mean_sac, std_sac, x)
            # p = gaussian(mean_sac, std_sac, x)
            # if p == 0:
            #     p = 0.0001
            # Sac[o] = math.log(p)
        if o not in Fix:
            Fix[o] = gaussian(mean_fix, std_fix, x)
            # p = gaussian(mean_fix, std_fix, x)
            # if p == 0:
            #     p = 0.0001
            # Fix[o] = math.log(p)
    # normalize
    #Sac = normalize(Sac)
    #Fix = normalize(Fix)
    emit_p = {"Sac": Sac, "Fix": Fix}
    df['hidden_state'] = viterbi.run(obs, states, start_p, trans_p, emit_p)
    #print(len(df['hidden_state']))
    print(df['hidden_state'].value_counts())

    print('=============== END VITERBI ===============')

    # Q's: Why is probability so small? Should I be working with logs?


    print('\n============== BEGIN BAUM-WELCH ==============')

    # re-format to comply with baum-welch function
    # turn trans_p from dict to np array
    transition = []
    for d in trans_p.values():
        transition.append(dict_to_list(d))
    transition = np.array(transition)

    # turn emit_p from dict to np array
    emission = []
    for d in emit_p.values():
        emission.append(dict_to_list(d))
    emission = np.array(emission)

    start_probs = dict_to_list(start_p)

    #baum_welch.run(obs, states, start_probs, transition, emission)

    print('============== END BAUM-WELCH ==============')
"""

if __name__ == "__main__":
    # Testing
    # hello("Isabella")
    main()
