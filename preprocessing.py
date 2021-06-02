# import sys
# sys.path.insert(1, "/Users/ischoning/PycharmProjects/eyeMovements/eventdetect-master")
from pylab import *
import numpy as np
from scipy import signal
import pandas as pd


def format(file):

    # for file == 'varjo_events_12_0_0.txt' and the like
    if 'varjo_events' in file:
        df = pd.read_csv(file, sep="\t", float_precision=None)
        df = df[['time', 'device_time', 'left_pupil_measure1', 'right_pupil_measure1', 'target_angle_x',
                 'target_angle_y', 'left_angle_x', 'left_angle_y', 'right_angle_x', 'right_angle_y']]

    # for file == 'data_collection_events_eyetracker_MonocularEyeSampleEvent.csv' and the like
    elif 'eyelink' in file:
        df = pd.read_csv(file, sep=';', float_precision=None)
        df = df[['time', 'pupil_measure1', 'gaze_x', 'gaze_y', 'ppd_x', 'ppd_y']]
        df['angle_x'] = df.gaze_x / df.ppd_x # gaze is in pixels
        df['angle_y'] = df.gaze_y / df.ppd_y

    # for file == 'participant08_preprocessed172.csv' and the like
    elif 'participant' in file and 'preprocessed' in file:
        df = pd.read_csv(file)
        df = df[['timestamp_milis', 'raw_timestamp', 'left_pupil_size', 'right_pupil_size',
                 'degrees_LEFT_horizontal', 'degrees_LEFT_vertical',
                 'degrees_RIGHT_horizontal', 'degrees_RIGHT_vertical']]
        df['timestamp_milis'] = df['timestamp_milis']/1000.0
        df = df.rename(columns={'timestamp_milis':'time', 'raw_timestamp':'device_time',
                                'left_pupil_size':'left_pupil_measure1','right_pupil_size':'right_pupil_measure1',
                                'degrees_LEFT_horizontal':'left_angle_x', 'degrees_LEFT_vertical':'left_angle_y',
                                'degrees_RIGHT_horizontal':'right_angle_x','degrees_RIGHT_vertical':'right_angle_y'})
    else:
        raise Exception('The reformatting for this file has not been written yet.')

    return df


def get_feats(df, eye):
    if 'right' in eye or 'Right' in eye:
        x = df['right_angle_x']
        y = df['right_angle_y']
        v = df['vel_r']
        a = df['accel_r']
    elif 'left' in eye or 'Left' in eye:
        x = df['left_angle_x']
        y = df['left_angle_y']
        v = df['vel_l']
        a = df['accel_l']
    elif eye == 'monocular':
        x = df['angle_x']
        y = df['angle_y']
        v = df['vel']
        a = df['accel']

    return x, y, v, a

def calculate_features(df, smooth = False):
    if smooth:
        # Smooth using kernel windows of sizes 3, 5, 7
        window_size = 7
        # if monocular
        if 'ppd_x' in df.columns:
            data = (df.angle_x, df.angle_y)
        # else binocular
        else:
            data = (df.left_angle_x, df.left_angle_y, df.right_angle_x, df.right_angle_y)
        for i in range(len(df) - window_size):
            for elem in data:
                # if first and last elements in window have less than 0.5 deg difference
                if abs(elem[i] - elem[i + window_size - 1]) < 0.5:
                    m = (elem[i] + elem[i + window_size - 1]) / 2.0
                    # set all elements in between equal to their average
                    for j in range(i + 1, i + window_size - 1):
                        elem[j] = m

        if 'ppd_x' in df.columns:
            df.loc[:, 'angle_x'] = signal.savgol_filter(df.loc[:, 'angle_x'], window_length=101, polyorder=3)
            df.loc[:, 'angle_y'] = signal.savgol_filter(df.loc[:, 'angle_y'], window_length=101, polyorder=3)
        else:
            df.loc[:, 'left_angle_x'] = signal.savgol_filter(df.loc[:, 'left_angle_x'], window_length=101, polyorder=3)
            df.loc[:, 'left_angle_y'] = signal.savgol_filter(df.loc[:, 'left_angle_y'], window_length=101, polyorder=3)
            df.loc[:, 'right_angle_x'] = signal.savgol_filter(df.loc[:, 'right_angle_x'], window_length=101, polyorder=3)
            df.loc[:, 'right_angle_y'] = signal.savgol_filter(df.loc[:, 'right_angle_y'], window_length=101, polyorder=3)

    # calculate intersample time
    df.loc[:, 'isi'] = np.diff(df.time, prepend=0)
    # if isi == 0.0, replace with average isi
    df.loc[:, 'isi'] = np.where(df.isi == 0.0, np.average(df.isi), df.isi)

    # if monocular
    if 'ppd_x' in df.columns:
        #df.loc[:,'d'] = sqrt((df.angle_x + df.angle_y)**2)
        del_x = np.diff(df['angle_x'], prepend=0)
        del_y = np.diff(df['angle_y'], prepend=0)
        # df.loc[:,'d'] = del_x + del_y
        # if smooth:
        #     df.loc[:, 'd'] = signal.savgol_filter(df.loc[:, 'd'], window_length=51, polyorder=3)
        # del_d = np.diff(df['d'], prepend=0)
        #df.loc[:,'vel'] = abs(del_d) / df['isi']
        df.loc[:,'vel'] = abs(del_x)/df['isi'] + abs(del_y)/df['isi']
        if smooth:
            df.loc[:,'vel'] = signal.savgol_filter(df.loc[:, 'vel'], window_length=101, polyorder=3)
        del_vel = np.diff(df['vel'], prepend=0)
        df.loc[:,'accel'] = abs(del_vel) / df['isi']
        if smooth:
            df.loc[:, 'accel'] = signal.savgol_filter(df.loc[:, 'accel'], window_length=101, polyorder=3)
        del_accel = np.diff(df['accel'], prepend=0)
        df.loc[:,'jolt'] = del_accel / df['isi']
        # df.loc[:,'del_d'] = del_d

    # if binocular
    else:
        # df.loc[:, 'd_r'] = abs(df.right_angle_x) + abs(df.right_angle_y)
        # df.loc[:, 'd_l'] = abs(df.left_angle_x) + abs(df.left_angle_y)
        del_x_r = np.diff(df['right_angle_x'], prepend=0)
        del_y_r = np.diff(df['right_angle_y'], prepend=0)
        del_x_l = np.diff(df['left_angle_x'], prepend=0)
        del_y_l = np.diff(df['left_angle_y'], prepend=0)

        # if smooth:
        #     df.loc[:, 'd_r'] = signal.savgol_filter(df.loc[:, 'd_r'], window_length=51, polyorder=3)
        #     df.loc[:, 'd_l'] = signal.savgol_filter(df.loc[:, 'd_l'], window_length=51, polyorder=3)
        #
        # del_d_r = np.diff(df['d_r'], prepend=0)
        # del_d_l = np.diff(df['d_l'], prepend=0)

        # df.loc[:, 'vel_r'] = abs(del_d_r) / df['isi']
        # df.loc[:, 'vel_l'] = abs(del_d_l) / df['isi']
        df.loc[:,'vel_r'] = abs(del_x_r)/df['isi'] + abs(del_y_r)/df['isi']
        df.loc[:, 'vel_l'] = abs(del_x_l) / df['isi'] + abs(del_y_l) / df['isi']

        if smooth:
            df.loc[:, 'vel_r'] = signal.savgol_filter(df.loc[:, 'vel_r'], window_length=101, polyorder=3)
            df.loc[:, 'vel_l'] = signal.savgol_filter(df.loc[:, 'vel_l'], window_length=101, polyorder=3)

        del_vel_r = np.diff(df['vel_r'], prepend=0)
        del_vel_l = np.diff(df['vel_l'], prepend=0)

        df.loc[:, 'accel_r'] = abs(del_vel_r) / df['isi']
        df.loc[:, 'accel_l'] = abs(del_vel_l) / df['isi']

        if smooth:
            df.loc[:, 'accel_r'] = signal.savgol_filter(df.loc[:, 'accel_r'], window_length=101, polyorder=3)
            df.loc[:, 'accel_l'] = signal.savgol_filter(df.loc[:, 'accel_l'], window_length=101, polyorder=3)

        del_accel_r = np.diff(df['accel_r'], prepend=0)
        del_accel_l = np.diff(df['accel_l'], prepend=0)

        df.loc[:, 'jolt_r'] = del_accel_r / df['isi']
        df.loc[:, 'jolt_l'] = del_accel_l / df['isi']

        # df.loc[:, 'del_d_r'] = del_d_r
        # df.loc[:, 'del_d_l'] = del_d_l

    # remove the first three datapoints (due to intersample calculations)
    df = df[3:]
    df.reset_index(drop=True, inplace=True)

    return df


def remove_outliers(df):
    """
    :param df: a dataframe with the below columns
    :return: a dataframe cleaned from outliers
    """
    print("Preprocessing....")

    ### STEP 0: Downsample the data, remove NAN, calculate features and target ###

    len_init = len(df)

    # sample rate: 100 samples per sec
    # downsample (always remove the first few seconds of recording)
    # BLOCK 'sp' [0:91913] ; BLOCK 'fs' from [~100000:151280]
    if len(df >= 7000):
        #df = df[1000:int(91913/8)]
        df = df[120000:130000]
    downsample = len(df)

    # Remove NA's
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna()
    df.reset_index(drop=True, inplace=True)
    len_na = downsample - len(df)

    # Calculate amplitude, velocity, acceleration, change in acceleration
    df = calculate_features(df, smooth=False)
    drop = 3

    # remove one-sample spikes: use a median filter of length 3 iff the point satisfies the amplitude and velocity criteria
    a_min = 0.3    # [deg] min amplitude for a one-sample spike (Table 3 in Larsson etal 2013)

    ### STEP 1: Clean data based on eye physiology ###

    bad_data = [] # list of indices
    cond_1, cond_2, cond_3, cond_4, cond_5 = 0, 0, 0, 0, 0

    # if monocular
    if 'ppd_x' in df.columns:

        for i in range(1, len(df)-2):

            # no pupil size?
            if df.loc[i,'pupil_measure1'] == 0:
                bad_data.append(i)
                cond_1 += 1

            # remove negative velocity (should be abs)
            elif df.loc[i,'vel'] < 0:
                bad_data.append(i)
                cond_2 += 1

            # angular velocity greater than 1000 deg/s?
            elif df.loc[i,'vel'] > 1000:
                bad_data.append(i)
                cond_3 += 1

    # else if binocular
    else:
        for i in range(1, len(df)-2):

            # no pupil size?
            if df.loc[i,'left_pupil_measure1'] == 0 or df.loc[i,'right_pupil_measure1'] == 0:
                bad_data.append(i)
                cond_1 += 1

            # remove negative velocity (should be abs)
            elif df.loc[i,'vel_l'] < 0 or df.loc[i,'vel_r'] < 0:
                bad_data.append(i)
                cond_2 += 1

            # angular velocity greater than 1000 deg/s?
            elif df.loc[i,'vel_l'] > 1000 or df.loc[i,'vel_r'] > 1000:
                bad_data.append(i)
                cond_3 += 1

    df.drop(index=bad_data, inplace=True)
    df.reset_index(drop=True, inplace=True)


    ### STEP 2: Smooth the clean data ###

    df = calculate_features(df, smooth = True)
    drop += 3

    # Calculate target (ignore the first three datapoints)
    if 'target_angle_x' in df.columns:
        df.loc[3:,'target_d'] = df.loc[3:,'target_angle_x'] + df.loc[3:,'target_angle_y']
        del_target_d = np.diff(df['target_d'], prepend=0)
        df.loc[:,'target_vel'] = abs(del_target_d) / df['isi']
        for i in range(1,len(df)-1):
            prev = df.target_vel[i-1]
            next = df.target_vel[i+1]
            if df.target_vel[i] == 0 and prev != 0 and next != 0:
                df.target_vel[i] = abs(next+prev)/2.0
        del_target_vel = np.diff(df.target_vel, prepend=0)
        df.loc[3:,'target_accel'] = abs(del_target_vel) / df['isi']

    print("================= PREPROCESSING RESULTS =====================")
    print("len of dataset before processing:", len_init)
    print("len of downsampled dataset:", downsample)
    print("Number of inf and na removed:", len_na)
    print("Ends dropped (due to difference calculations):", drop)

    print("\nNumber of datapoints with no pupil size:", cond_1)
    print("Negative intersample velocity:", cond_2)
    print("Intersample velocity greater than 1000 deg/s:", cond_3)

    print("\nlen of 'bad data':", len(bad_data))
    print("len of dataset after cleaning:", len(df))

    print("=============================================================")


    return df