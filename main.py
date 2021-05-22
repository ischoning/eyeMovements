# import sys
# sys.path.insert(1, "/Users/ischoning/PycharmProjects/eyeMovements/eventdetect-master")
import pandas as pd
from pylab import *
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import math
import viterbi
import baum_welch
from preprocessing import remove_outliers
from preprocessing import get_feats
import plots
from detect.dispersion import Dispersion
from detect.sample import Sample
from detect.sample import ListSampleStream


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


def print_events(df):
    events = []
    event_label = df.event.iloc[0]
    start = df.time.iloc[0]
    n = 1
    event_change = False
    for i in range(len(df)):
        if i == len(df)-1 or df.event.iloc[i] != df.event.iloc[i+1]:
            event_change = True
        if event_change:
            end = df.time.iloc[i]
            events.append([event_label, n, start, end])
            if i != len(df)-1:
                start = df.time.iloc[i+1]
                event_label = df.event.iloc[i+1]
                n = 1
                event_change = False
        else:
            n += 1

    print('=====================================================================')

    for e in events:
        print("%s for %d samples from %d ms to %d ms." % (e[0], e[1], e[2], e[3]))

    print('=====================================================================')

    print('Number of Fixation events:', sum(df.event=='fix'))
    print('Number of Smooth Pursuit events:', sum(df.event=='smp'))
    print('Number of Saccade events:', sum(df.event == 'sac'))
    print('Number of Other events:', sum(df.event == 'other'))

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
    file = '/Users/ischoning/PycharmProjects/GitHub/data/varjo_events_5_0_0.txt'

    # create dataframe
    df = pd.read_csv(file, sep="\t", float_precision=None)
    df = df[['time', 'dt', 'device_time', 'left_pupil_measure1', 'right_pupil_measure1', 'target_angle_x', 'target_angle_y',
         'right_gaze_x', 'right_gaze_y', 'left_gaze_x', 'left_gaze_y', 'left_angle_x', 'left_angle_y', 'right_angle_x', 'right_angle_y']]

    # select eye data to analyze ('left' or 'right')
    eye = 'left'


    #### STEP 1: Clean Outliers ####
    df = remove_outliers(df)

    # instantiate data according to eye, selected above
    pix_x, pix_y, x, y, d, v, a, del_d = get_feats(df, eye)
    df['d'] = d
    df['v'] = v
    df['a'] = a
    df['del_d'] = del_d
    df['pix_x'] = pix_x
    df['pix_y'] = pix_y

    #df['v'] = np.convolve(df.vel_r, df.vel_l, mode='same')/(2*len(df))
    #df['a'] = np.convolve(df.accel_r, df.accel_l, mode='same')

    # plot results after removing outliers
    plots.plot_path(df)
    plots.plot_vs_time(df, feat = d, label ='Amplitude', eye = eye)
    plots.plot_vs_time(df, feat = v, label ='Velocity', eye = eye)
    #plots.plot_vs_time(df, feat = df.a, label = 'Acceleration', eye=eye)


    #### STEP 2: Filter Fixations Using Dispersion (I-DT) Algorithm ####

    # select threshold method (Velocity or Dispersion)
    method = 'Dispersion'
    window_sizes = (5, 10,15,20)
    threshes = (40.4,60.6) # 40.4 pixels in 1 deg (overleaf doc sacVelocity.py)
    # using 1 deg from Pieter Blignaut's paper: Fixation identification: "The optimum threshold for a dispersion algorithm"

    fig, ax = plt.subplots(len(threshes),len(window_sizes), figsize=(16,10))

    nrow = 0
    ncol = 0
    for window_size in window_sizes:
        for thresh in threshes:
            samples = [Sample(ind=i, time=df.time[i], x=df.pix_x[i], y=df.pix_y[i]) for i in range(len(df))]
            stream = ListSampleStream(samples)
            fixes = Dispersion(sampleStream = stream, windowSize = window_size, threshold = thresh)
            centers = []
            num_samples = []
            starts = []
            ends = []
            for f in fixes:
                centers.append(f.get_center())
                num_samples.append(f.get_num_samples())
                starts.append(f.get_start())
                ends.append(f.get_end())
                print(f)
            print("Number of fix events:", len(centers))
            print("Number of fix samples:", np.sum(num_samples))

            # label the fixations in the dataframe
            df['event'] = 'other'
            count = 0
            print('len(centers):', len(centers))
            for i in range(len(starts)):
                df.loc[starts[i]:ends[i], ("event")] = 'fix'
                # if the end of the data is all fixations
                if i == len(starts)-1:
                    df.loc[starts[i]:len(starts), ("event")] = 'fix'
                # if there are only 1 or 2 samples between fixations, combine them
                elif starts[i+1]-ends[i] <= 2:
                    count += 1
                    df.loc[ends[i]:starts[i+1], ("event")] = 'fix'
            print(count)

            centers = np.array(centers)
            ax[nrow][ncol].scatter(df.right_angle_x[df.event !='fix'], df.right_angle_y[df.event!='fix'], s=0.5,label='other')
            ax[nrow][ncol].scatter(df.right_angle_x[df.event =='fix'], df.right_angle_y[df.event =='fix'], color='r', s=0.5, label='fix')
            # for i in range(len(centers)):
            #     plots.circle(centers[i], radius=num_samples[i]*0.5+10)
            #plt.scatter(centers[:,0], centers[:,1], c='None', edgecolors='r')
            ax[nrow][ncol].set_title('Window: '+str(window_size)+' Thresh: '+str(thresh))
            ax[nrow][ncol].set_xlabel('x pixel')
            ax[nrow][ncol].set_ylabel('y pixel')
            ax[nrow][ncol].legend()

            nrow += 1
        nrow = 0
        ncol += 1
    plt.legend()
    plt.show()
    #
    # centers = np.array(centers)
    # plt.scatter(pix_x[df.event !='fix'], pix_y[df.event!='fix'], s=0.5)
    # plt.scatter(pix_x[df.event =='fix'], pix_y[df.event =='fix'], color='r', s=0.5)
    # plt.title('Fixations using WindowSize '+str(window_size)+' and Thresh '+str(thresh))
    # plt.xlabel('x pixel')
    # plt.ylabel('y pixel')
    # plt.show()

    # # set variables
    # if method == 'Dispersion':
    #     var = np.abs(del_d)
    #     x_axis = 'deg'
    # elif method == 'Velocity':
    #     var = v
    #     x_axis = 'deg/s'
    #
    # # plot histogram
    # head = eye + ' eye: ' + method
    # hist, bin_edges = plots.plot_hist(df, method = method, eye = eye, title=head, x_axis=x_axis)
    #
    # # sanity check:
    # prob = hist/len(hist)
    # print('sum of hist:', np.sum(prob))
    #
    # # calculate distribution characteristics
    # mu = np.mean(var)
    # sigma = np.std(var)
    # med = np.median(var)
    # print("mean:", mu, "std:", sigma)
    # print("median:", med)
    #
    # # set initial threshold values
    # if method == 'Dispersion':
    #     thresh_init = 1  # degree
    # elif method == 'Velocity':
    #     thresh_init = mu
    #
    # # create fixation gaussian
    # fix = var[var <= thresh_init]
    # mu_fix = np.mean(fix)
    # sigma_fix = np.std(fix)
    # med_fix = np.median(fix)
    # print("fix mean:", mu_fix, "fix std:", sigma_fix)
    # print("fix median:", med_fix)
    #
    # # update threshold values
    # if method == 'Dispersion':
    #     thresh = 1  # degree
    # elif method == 'Velocity':
    #     thresh = mu_fix + 3 * sigma_fix
    #
    # # basic threshold classification
    # df['event'] = np.where(var <= thresh, 'Fix', 'Sac')
    #
    # # plot classification
    # plots.plot_events(df, eye =eye)
    #
    #
    # #### Step 3: Filter Saccades Using Velocity (I-VT) Algorithm ####
    #
    # # select threshold method (Velocity or Dispersion)
    # method = 'Velocity'
    #
    # # set variables
    # if method == 'Dispersion':
    #     var = np.abs(del_d)
    #     x_axis = 'deg'
    # elif method == 'Velocity':
    #     var = v
    #     x_axis = 'deg/s'
    #
    # # plot histogram
    # head = eye + ' eye: ' + method
    # hist, bin_edges = plots.plot_hist(df, method=method, eye=eye, title=head, x_axis=x_axis)
    #
    # # sanity check:
    # prob = hist / len(hist)
    # print('sum of hist:', np.sum(prob))
    #
    # # calculate distribution characteristics
    # mu = np.mean(var)
    # sigma = np.std(var)
    # med = np.median(var)
    # print("mean:", mu, "std:", sigma)
    # print("median:", med)
    #
    # # set initial threshold values
    # if method == 'Dispersion':
    #     thresh_init = 1  # degree
    # elif method == 'Velocity':
    #     thresh_init = mu
    #
    # # create non-fixation gaussian
    # sac = var[var > thresh_init]
    # mu_sac = np.mean(sac)
    # sigma_sac = np.std(sac)
    # med_sac = np.median(sac)
    # print("non-fix mean:", mu_sac, "non-fix std:", sigma_sac)
    # print("non-fix median:", med_sac)
    #
    # # update threshold values
    # if method == 'Dispersion':
    #     thresh = 1  # degree
    # elif method == 'Velocity':
    #     thresh = mu
    #
    # # basic threshold classification
    # df['event2'] = np.where(var > thresh, 'Sac', 'SmP')
    # df.event = np.where(df.event != 'Fix', df.event2, 'Fix')
    #
    # # plot classification
    # plots.plot_events(df, eye=eye)
    #
    #
    # #### STEP 4: Filter Saccades Using Carpenter's Theorem ####
    #
    # # create sequence of events
    # seq = pd.DataFrame(sequence(df))
    #
    # # plot amplitude and velocity of "Sac's" along with ideal (Carpenter's: D = 21 + 2.2A, D~ms, A~deg)
    # non_fix = seq[seq.State == 'Sac']
    # plt.scatter(non_fix.Amplitude, non_fix.Duration_ms, label = "'non-fixation' samples")
    # x = linspace(min(non_fix.Amplitude),max(non_fix.Amplitude))
    # y = 21 + 2.2*x     # Carpenter's Theorem
    # plt.plot(x, y, color = 'green', label = 'D = 21 + 2.2A')
    # head = '[' + eye + ' eye] Saccades: Amplitude vs Duration'
    # plt.title(head)
    # plt.xlabel('amplitude (deg)')
    # plt.ylabel('duration (ms)')
    # plt.legend()
    # plt.show()
    #
    # print("========= FULL SEQUENCE =========")
    # print(seq)
    # print("========= SAC SEQUENCE =========")
    # print(non_fix)
    #
    # # calculate error rate
    # df['error'] = (abs(non_fix.Duration_ms - (21+2.2*non_fix.Amplitude))/(21+2.2*non_fix.Amplitude))*100
    # print("Average Error:", np.mean(df.error), "%")
    #
    # # classify "Sac" into "Sac" or "Other" depending on error rate with Carpenter's Theorem

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
