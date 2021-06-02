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
import preprocessing
import plots
from detect.dispersion import Dispersion
from detect.sample import Sample
from detect.sample import ListSampleStream
from detect.velocity import Velocity
from detect.intersamplevelocity import IntersampleVelocity
import statistics


def clean_sequence(df, sequence):
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
        #print(f)

    # label the fixations in the dataframe
    df['event'] = 'other'
    count = 0
    for i in range(len(starts)):
        df.loc[starts[i]:ends[i], ('event')] = 'fix'
        # if the end of the data is all fixations
        if i == len(starts) - 1:
            df.loc[starts[i]:len(starts), ('event')] = 'fix'
        # if there are only 1 or 2 samples between fixations, combine them
        elif starts[i + 1] - ends[i] <= 2:
            count += 1
            df.loc[ends[i]:starts[i + 1], ('event')] = 'fix'

    # plot classification
    plots.plot_vs_time(df, label='', eye=eye, classify=True, method=method)


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


def sequence(df):
    # determine the sequence of events (ignore the first element)

    prev_st = df.event[1]
    t0 = df.time[1]
    x0 = df.x[1]
    y0 = df.y[1]
    start = 1
    sequence = []
    count = 1
    for i in range(2, len(df)):
        st = df.event[i]
        if st == prev_st:
            count += 1
        elif st != prev_st:
            t = (df.time[i-1] - t0)*1000
            amp = np.sqrt((x0-df.x[i-1])**2 + (y0-df.y[i-1])**2)
            end = i
            sequence.append({'State':prev_st, 'Num_samples':count, 'Amplitude':amp, 'Duration_ms':t, 'start':start, 'end':end})
            count = 1
            prev_st = st
            t0 = df.time[i]
            x0 = df.x[i]
            y0 = df.y[i]
            start = i
        if i == len(df) - 1:
            t = (df.time[i] - t0) * 1000
            amp = np.sqrt((x0-df.x[i-1])**2 + (y0-df.y[i-1])**2)
            end = i+1
            sequence.append({'State':prev_st, 'Num_samples':count, 'Amplitude':amp, 'Duration_ms':t, 'start':start, 'end':end})

    # print("Sequence:", sequence)

    return sequence


def find_mode(feat):
    """
    :param feat: intersample velocity
    :return: the mode (if found up to 3 decimal points), otherwise none
    """
    decimal = 1
    while decimal <= 3:
        try:
            return statistics.mode(np.round(feat, decimal))
        except:
            decimal += 1
    if decimal == 4:
        return "More than one mode using up to 3 decimal points."


def main():

    #### STEP 0: Load Data and Set Constants ####
    #
    # upload file
    #
    file = '/Users/ischoning/PycharmProjects/GitHub/data/varjo_events_12_0_0.txt'
    file = '/Users/ischoning/PycharmProjects/GitHub/data/hdf5/events_1_eyelink/data_collection_events_eyetracker_MonocularEyeSampleEvent.csv'
    #
    # create dataframe
    #
    df = preprocessing.format(file)
    #
    # set constants depending on file type (varjo or eyelink)
    #
    if 'eyelink' in file:
        eye = 'monocular'
        #
        # set thresholds and window sizes for testing
        #
        window_sizes = (100, 125, 150) # number of samples (ms)
        threshes = (1, 1.5) # 40.4 pixels in 1 deg (overleaf doc sacVelocity.py)  # using 1 deg from Pieter Blignaut's paper: Fixation identification: "The optimum threshold for a dispersion algorithm"
        #
        # set best dispersion threshold and window size
        #
        best_window_size = 150
        best_thresh = 1
        # this means that in order to be classified as a fixation, dispersion
        # should not go above 1 degree for at least 100 ms (which corresponds
        # to 100 samples for eyelink data at 1000Hz sample rate)
        #
    else:
        #
        # choose eye data to analyze {'left eye' or 'right eye'}
        #
        eye = 'left eye'
        #
        # set thresholds and window sizes for testing
        #
        window_sizes = (15, 20, 25) # number of samples
        threshes = (0.5, 1.0) # 40.4 pixels in 1 deg (overleaf doc sacVelocity.py)
        #
        # set best dispersion threshold and window size
        #
        best_window_size = 20
        best_thresh = 0.5
        # this means that in order to be classified as a fixation, dispersion
        # should not go above 0.5 degrees for at least 200 ms (which corresponds
        # to 20 samples for varjo data at 100Hz sample rate)
        #


    #### STEP 1: Clean Outliers ####
    #
    df = preprocessing.remove_outliers(df)
    #
    # instantiate data according to eye, selected above
    #
    x, y, v, a = preprocessing.get_feats(df, eye)
    df['x'] = x
    df['y'] = y
    df['v'] = v
    df['a'] = a
    #
    # plot results after removing outliers
    #
    # scatter plot of movement (x vs y) in degrees
    plots.plot_path(df)
    # behavior (x, y, velocity over time)
    plots.plot_vs_time(df, label ='Velocity', eye = eye)


    #### STEP 2: Filter Fixations Using Dispersion (I-DT) Algorithm ####
    #
    # plot variations of window_size and threshold
    # best window size and threshold set in step 0
    #
    plots.plot_IDT_thresh_results(df.copy(),window_sizes,threshes) # threshes defined in Step 0
    #
    # classify fixations using I-DT
    #
    df = label_fixes(df.copy(), eye=eye, ws=best_window_size, thresh=best_thresh, method = 'IDT')
    #
    # show sequence of events
    #
    seq = pd.DataFrame(sequence(df))
    fix = seq[seq.State == 'fix']
    print("================= I-DT RESULTS =====================")
    print("Fix Duration_ms < window (150ms):", np.sum(np.where(fix.Duration_ms < 150, 1, 0)))
    print("Fix Amplitude > thresh (1 deg):", np.sum(np.where(fix.Amplitude > 1, 1, 0)))
    print("Fix Sequence:")
    print(fix)
    print("=============================================================")


    #### STEP 3A: Filter Saccades Using Velocity Threshold ####
    #
    # find modal intersample velocity for fixations (if more than one mode, use mean)
    # (round to 1 decimal first)
    #
    try:
        fix_mode_v = statistics.mode(np.round(df[df.event == 'fix'].v, 1))
        kw = 'modal'
    except:
        fix_mode_v = np.average(df[df.event == 'fix'].v)
        kw = 'mean'
    print(kw, "intersample velocity for fixations:", fix_mode_v)
    #
    # take margin of error above the mode as velocity based threshold
    #
    error = abs(df[df.v > fix_mode_v].v - fix_mode_v) / fix_mode_v
    print("total margin of error above fix_mode_v:", np.average(error))
    thresh = fix_mode_v * np.average(error) * 2
    #
    # compare to margin of error within fixations
    #
    error_fix = abs(df[df.event == 'fix'].v - fix_mode_v) / fix_mode_v
    print("fix margin of error:", np.average(error_fix))
    #
    # classify as saccade if intersample velocity > fix_mode_v * 2 * error
    #
    df_copy = df.copy()
    df_copy['event'] = np.where(df_copy.v > thresh, 'sac', df_copy.event)
    #
    # relabel the remaining 'other' as smooth pursuit
    #
    df_copy['event'] = np.where(df_copy.event == 'other', 'smp', df_copy.event)
    #
    # plot resulting classification
    #
    plots.plot_vs_time(df_copy, label='Velocity', eye=eye, classify=True, method='IVT')
    #
    # # saccade if intersample velocity > 22 deg/s (Houpt)
    # df_copy = df.copy()
    # df_copy['event'] = np.where(df_copy.v > 22, 'sac', df_copy.event)
    # # relabel the remaining 'other' as smooth pursuit
    # df_copy['event'] = np.where(df_copy.event == 'other', 'smp', df_copy.event)
    # # plots.plot_events(df_copy,eye,'IVT')
    # plots.plot_vs_time(df_copy, label='Velocity', eye=eye, classify=True, method='IVT')
    #
    # print event sequence results
    #
    seq = pd.DataFrame(sequence(df_copy))
    fix = seq[seq.State == 'fix']
    smp = seq[seq.State == 'smp']
    sac = seq[seq.State == 'sac']
    print("================= I-VT RESULTS =====================")
    print("Num Fix Events:", len(fix))
    print("Num SmP Events:", len(smp))
    print("Num Sac Events:", len(sac))
    print("=============================================================")
    print("Fix Sequence (Carpenter):")
    print(fix)
    print("SmP Sequence (Carpenter):")
    print(smp)
    print("Sac Sequence (Carpenter):")
    print(sac)
    print("=============================================================")


    #### STEP 3B: Filter Saccades Using Carpenter's Theorem ####
    #
    # create sequence of events
    seq = pd.DataFrame(sequence(df))
    #
    # plot amplitude and velocity of non-fixes along with ideal from Carpenter's Thm: D = 21 + 2.2A, [D~ms, A~deg]
    #
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
    #
    # calculate error rate
    #
    seq['error'] = abs(seq.Duration_ms - (21 + 2.2 * seq.Amplitude)) / (21 + 2.2 * seq.Amplitude)
    #
    # classify other into saccade or smooth pursuit depending on error rate with Carpenter's Theorem
    # If actual is greater than 10% of true then classify as smp, otherwise sac.
    #
    seq['State'] = np.where(seq.State == 'other', np.where(seq.error < 0.2, 'sac', seq.State),seq.State)
    #
    # relabel the remaining 'other' as smooth pursuit
    #
    seq['State'] = np.where(seq.State == 'other', 'smp', seq.State)
    #
    # remap seq State to the dataframe df
    #
    for i in range(len(seq)):
        df.loc[seq.start[i]:seq.end[i],'event'] = seq.State[i]
    #
    # plot result
    #
    plots.plot_vs_time(df, label='Velocity', eye=eye, classify=True, method='Carpenter')
    #
    # print event sequence results
    #
    fix = seq[seq.State == 'fix']
    smp = seq[seq.State == 'smp']
    sac = seq[seq.State == 'sac']
    print("================= CARPENTER RESULTS =====================")
    print("Num Fix Events:", len(fix))
    print("Num SmP Events:", len(smp))
    print("Num Sac Events:", len(sac))
    print("=============================================================")
    print("Fix Sequence (Carpenter):")
    print(fix)
    print("SmP Sequence (Carpenter):")
    print(smp)
    print("Sac Sequence (Carpenter):")
    print(sac)
    print("=============================================================")


    #### STEP 4: Plot resulting histograms and statistics ####
    #
    # output final stats
    #
    print("==== CLASSIFICATION RESULTS (I-DT followed by Carpenter) ====")
    print("average intersample velocity:")
    print("    raw:", np.round(np.average(df.v),3))
    print("    fix:", np.round(np.average(df[df.event == 'fix'].v),3))
    print("    smp:", np.round(np.average(df[df.event == 'smp'].v),3))
    print("    sac:", np.round(np.average(df[df.event == 'sac'].v),3))
    #
    print("modal intersample velocity:")
    print("    raw:", find_mode(df.v))
    print("    fix:", find_mode(df[df.event == 'fix'].v))
    print("    smp:", find_mode(df[df.event == 'smp'].v))
    print("    sac:", find_mode(df[df.event == 'sac'].v))
    #
    print("velocity standard deviation:")
    print("    raw:", np.round(df.v.std(),3))
    print("    fix:", np.round(df[df.event == 'fix'].v.std(),3))
    print("    smp:", np.round(df[df.event == 'smp'].v.std(),3))
    print("    sac:", np.round(df[df.event == 'sac'].v.std(),3))
    #
    # plot velocity histograms by classification
    #
    plots.plot_vel_hist(df.copy(), eye=eye, title='Velocity Histogram: All', density=True, classify=True)
    #
    # plot carpenter error histogram by classification
    #
    seq.loc[:,'error'] = np.round(seq.error, 1)
    num_bins = range(int(math.floor(np.min(seq.error))),int(math.ceil(np.max(seq.error))),1)
    common_params = dict(bins=num_bins,
                         color = ('green','orange','blue'),
                         label = ('fixation','smooth pursuit','saccade'),
                         alpha = 0.6,
                         density=True)
    plt.hist((fix.error, smp.error, sac.error), **common_params)
    plt.title('Carpenter Error')
    plt.legend()
    plt.ylabel('Frequency')
    plt.xlabel('Error')
    plt.show()
    #
    print("============ ERROR STATS ============")
    print("mean error:")
    print("    raw:", np.round(np.average(seq.error), 3))
    print("    fix:", np.round(np.average(fix.error), 3))
    print("    smp:", np.round(np.average(smp.error), 3))
    print("    sac:", np.round(np.average(sac.error), 3))
    #
    print("mode error:")
    print("    raw:", find_mode(seq.error))
    print("    fix:", find_mode(fix.error))
    print("    smp:", find_mode(smp.error))
    print("    sac:", find_mode(sac.error))
    #
    print("std error:")
    print("    raw:", np.round(seq.error.std(), 3))
    print("    fix:", np.round(fix.error.std(), 3))
    print("    smp:", np.round(smp.error.std(), 3))
    print("    sac:", np.round(sac.error.std(), 3))


if __name__ == "__main__":
    # Testing
    # hello("Isabella")
    main()
