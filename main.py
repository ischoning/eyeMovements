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
from detect.velocity import Velocity
from detect.intersamplevelocity import IntersampleVelocity


def clean_sequence(df):
    # determine the sequence of events
    prev_st = df.event[0]
    sequence = []
    count = 1
    for i in range(1, len(df)):
        st = df.event[i]
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

def label_fixes(df, eye, ws = 0, thresh = 0, method='IDT'):

    samples = [Sample(ind=i, time=df.time[i], x=df.x[i], y=df.y[i]) for i in range(len(df))]
    stream = ListSampleStream(samples)

    if method == 'IDT':
        fixes = Dispersion(sampleStream = stream, windowSize = ws, threshold = thresh)
    elif method == 'IVT':
        fixes = Velocity(sampleStream=IntersampleVelocity(stream), threshold=thresh)
    else:
        raise Exception('method should be either "IDT" or "IVT"')

    centers = []
    num_samples = []
    starts = []
    ends = []
    for f in fixes:
        centers.append(f.get_center())
        num_samples.append(f.get_num_samples())
        starts.append(f.get_start())
        ends.append(f.get_end())
        # print(f)
    #print("Number of fix events:", len(centers))
    #print("Number of fix samples:", np.sum(num_samples))

    # label the fixations in the dataframe
    df['event'] = 'other'
    count = 0
    #print('len(centers):', len(centers))
    for i in range(len(starts)):
        df.loc[starts[i]:ends[i], ('event')] = 'fix'
        # if the end of the data is all fixations
        if i == len(starts) - 1:
            df.loc[starts[i]:len(starts), ('event')] = 'fix'
        # if there are only 1 or 2 samples between fixations, combine them
        elif starts[i + 1] - ends[i] <= 2:
            count += 1
            df.loc[ends[i]:starts[i + 1], ('event')] = 'fix'
    # print(count)

    # # label the saccades using velocity threshold (22 deg/s according to Houpt)
    # df['event'] = np.where(df.v >= 22, 'sac', df.event)

    # plot classification
    plots.plot_events(df, eye=eye, method = method)

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
    start = 0
    sequence = []
    count = 1
    for i in range(1, len(df)):
        st = df.event[i]
        if st == prev_st:
            count += 1
        elif st != prev_st:
            t = (df.time[i-1] - prev_t)*1000
            amp = abs(prev_amp - df.d[i-1])
            end = i
            sequence.append({'State':prev_st, 'Num_samples':count, 'Amplitude':amp, 'Duration_ms':t, 'start':start, 'end':end})
            count = 1
            prev_st = st
            prev_t = df.time[i]
            prev_amp = df.d[i]
            start = i
        if i == len(df) - 1:
            t = (df.time[i] - prev_t) * 1000
            amp = abs(prev_amp - df.d[i])
            end = i+1
            sequence.append({'State':prev_st, 'Num_samples':count, 'Amplitude':amp, 'Duration_ms':t, 'start':start, 'end':end})

    # print("Sequence:", sequence)

    return sequence


def main():

    #### STEP 0: Load Data and Set Constants ####
    # files
    file = '/Users/ischoning/PycharmProjects/GitHub/data/varjo_events_12_0_0.txt'

    # create dataframe
    df = pd.read_csv(file, sep="\t", float_precision=None)
    df = df[['time', 'dt', 'device_time', 'left_pupil_measure1', 'right_pupil_measure1', 'target_angle_x', 'target_angle_y',
         'right_gaze_x', 'right_gaze_y', 'left_gaze_x', 'left_gaze_y', 'left_angle_x', 'left_angle_y', 'right_angle_x', 'right_angle_y']]

    # select eye data to analyze ('left' or 'right')
    eye = 'left'

    # select method to use for fixation classification ('IVT' or 'IDT')
    fix_method_to_use = 'IDT'

    # select method to use for saccade and smooth pursuit classification ('Carpenter' or 'IVT')
    sac_method_to_use = 'IVT'


    #### STEP 1: Clean Outliers ####
    df = remove_outliers(df)

    # instantiate data according to eye, selected above
    pix_x, pix_y, x, y, d, v, a, del_d = get_feats(df, eye)
    df['pix_x'] = pix_x
    df['pix_y'] = pix_y
    df['x'] = x
    df['y'] = y
    df['d'] = d
    df['v'] = v
    df['a'] = a
    df['del_d'] = del_d

    #df['v'] = np.convolve(df.vel_r, df.vel_l, mode='same')/(2*len(df))
    #df['a'] = np.convolve(df.accel_r, df.accel_l, mode='same')

    # plot results after removing outliers
    plots.plot_path(df)
    plots.plot_vs_time(df, feat = d, label ='Amplitude', eye = eye)
    plots.plot_vs_time(df, feat = v, label ='Velocity', eye = eye)
    #plots.plot_vs_time(df, feat = df.a, label = 'Acceleration', eye=eye)


    #### STEP 2A: Filter Fixations Using Dispersion (I-DT) Algorithm ####

    # determine ideal window size and threshold
    window_sizes = (15,20,25,30)
    threshes = (0.50,0.75,1) # 40.4 pixels in 1 deg (overleaf doc sacVelocity.py)
    # using 1 deg from Pieter Blignaut's paper: Fixation identification: "The optimum threshold for a dispersion algorithm"
    plots.plot_fixations_IDT(df.copy(),window_sizes,threshes)
    best_window_size = 20
    best_thresh = 0.5 # in degrees
    if fix_method_to_use == 'IDT':
        df = label_fixes(df.copy(), eye=eye, ws=best_window_size, thresh=best_thresh, method = 'IDT')
    else:
        label_fixes(df.copy(), eye=eye, ws=best_window_size, thresh=best_thresh, method = 'IDT')


    #### STEP 2B: Filter Fixations Using Velocity (I-VT) Algorithm ####

    # run with optimal window size and threshold
    threshes = [1, 2, 3] # pixels per sample below which fixation
    # using 1 deg from Pieter Blignaut's paper: Fixation identification: "The optimum threshold for a dispersion algorithm"
    plots.plot_fixations_IVT(df.copy(),threshes)
    best_thresh = 3 # in degrees per sample
    if fix_method_to_use == 'IVT':
        df = label_fixes(df.copy(), eye=eye, thresh=best_thresh, method = 'IVT')
    else:
        label_fixes(df.copy(), eye=eye, thresh=best_thresh, method = 'IVT')


    #### STEP 3A: Filter Saccades Using Velocity Threshold ####
    # saccade if intersample velocity > 22 deg/s (Houpt)
    df_copy = df.copy()
    df_copy['event'] = np.where(df_copy.v > 22, 'sac', df_copy.event)
    # relabel the remaining 'other' as smooth pursuit
    df_copy['event'] = np.where(df_copy.event == 'other', 'smp', df_copy.event)
    plots.plot_events(df_copy,eye,'IVT')

    # print results
    seq = pd.DataFrame(sequence(df_copy))
    print("================= I-VT RESULTS =====================")
    print("Num Fix Events:", len(seq[seq.State == 'fix']))
    print("Num SmP Events:", len(seq[seq.State == 'smp']))
    print("Num Sac Events:", len(seq[seq.State == 'sac']))
    print("=============================================================")


    #### STEP 3B: Filter Saccades Using Carpenter's Theorem ####

    # create sequence of events
    df = df.copy()
    seq = pd.DataFrame(sequence(df))

    # plot amplitude and velocity of others along with ideal (Carpenter's: D = 21 + 2.2A, D~ms, A~deg)
    other = seq[seq.State == 'other']
    plt.scatter(other.Amplitude, other.Duration_ms, label = "other")
    x = linspace(min(other.Amplitude),max(other.Amplitude))
    y = 21 + 2.2*x     # Carpenter's Theorem
    plt.plot(x, y, color = 'green', label = 'D = 21 + 2.2A')
    head = '[' + eye + ' eye] Other: Amplitude vs Duration'
    plt.title(head)
    plt.xlabel('amplitude (deg)')
    plt.ylabel('duration (ms)')
    plt.legend()
    plt.show()

    # calculate error rate
    seq['error'] = (seq.Duration_ms - (21 + 2.2 * seq.Amplitude)) / (21 + 2.2 * seq.Amplitude)

    # classify other into saccade or smooth pursuit depending on error rate with Carpenter's Theorem
    seq['State'] = np.where(seq.State == 'other', np.where(seq.error < 0.1, 'sac', seq.State),seq.State)
    # relabel the remaining 'other' as smooth pursuit
    seq['State'] = np.where(seq.State == 'other', 'smp', seq.State)

    # remap seq State to the dataframe df
    for i in range(len(seq)):
        df.loc[seq.start[i]:seq.end[i],'event'] = seq.State[i]

    plots.plot_events(df, eye, 'Carpenter')

    sac = seq[seq.State == 'sac']

    # print results
    print("================= CARPENTER RESULTS =====================")
    print("Num Fix Events:", len(seq[seq.State == 'fix']))
    print("Num SmP Events:", len(seq[seq.State == 'smp']))
    print("Num Sac Events:", len(seq[seq.State == 'sac']))
    # calculate error between sac events and Carpenters Theorem
    print("Average Error:", np.round(abs(np.mean(sac.error)),3)*100, "%")
    print("=============================================================")

    # print("Full Sequence:")
    # print(seq)
    print("Sac Sequence (Carpenter):")
    print(sac)
    print("=============================================================")


if __name__ == "__main__":
    # Testing
    # hello("Isabella")
    main()
