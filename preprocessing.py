# import sys
# sys.path.insert(1, "/Users/ischoning/PycharmProjects/eyeMovements/eventdetect-master")
from pylab import *
import numpy as np
from scipy import signal


def get_feats(df, eye):
    if eye == 'right' or eye == 'Right':
        pix_x = df['right_gaze_x']
        pix_y = df['right_gaze_y']
        x = df['right_angle_x']
        y = df['right_angle_y']
        d = df['d_r']
        v = df['vel_r']
        a = df['accel_r']
        del_d = df['del_d_r']
    else:
        pix_x = df['left_gaze_x']
        pix_y = df['left_gaze_y']
        x = df['left_angle_x']
        y = df['left_angle_y']
        d = df['d_l']
        v = df['vel_l']
        a = df['accel_l']
        del_d = df['del_d_l']

    return pix_x, pix_y, x, y, d, v, a, del_d

def calculate_features(df, smooth = False):
    if smooth:
        # Smooth using kernel windows of sizes 3, 5, 7
        window_size = 7
        data = (df.left_angle_x, df.left_angle_y, df.right_angle_x, df.right_angle_y)
        for i in range(len(df) - window_size):
            for elem in data:
                # if first and last elements in window have less than 0.5 deg difference
                if abs(elem[i] - elem[i + window_size - 1]) < 0.5:
                    m = (elem[i] + elem[i + window_size - 1]) / 2.0
                    # set all elements in between equal to their average
                    for j in range(i + 1, i + window_size - 1):
                        elem[j] = m

        df.loc[:, 'left_angle_x'] = signal.savgol_filter(df.loc[:, 'left_angle_x'], window_length=101, polyorder=3)
        df.loc[:, 'left_angle_y'] = signal.savgol_filter(df.loc[:, 'left_angle_y'], window_length=101, polyorder=3)
        df.loc[:, 'right_angle_x'] = signal.savgol_filter(df.loc[:, 'right_angle_x'], window_length=101, polyorder=3)
        df.loc[:, 'right_angle_y'] = signal.savgol_filter(df.loc[:, 'right_angle_y'], window_length=101, polyorder=3)

    df.loc[:, 'isi'] = np.diff(df.time, prepend=0)

    df.loc[:, 'd_r'] = df.right_angle_x + df.right_angle_y
    df.loc[:, 'd_l'] = df.left_angle_x + df.left_angle_y

    if smooth:
        df.loc[:, 'd_r'] = signal.savgol_filter(df.loc[:, 'd_r'], window_length=51, polyorder=3)
        df.loc[:, 'd_l'] = signal.savgol_filter(df.loc[:, 'd_l'], window_length=51, polyorder=3)

    del_d_r = np.diff(df['d_r'], prepend=0)
    del_d_l = np.diff(df['d_l'], prepend=0)

    df.loc[:, 'vel_r'] = abs(del_d_r) / df['isi']
    df.loc[:, 'vel_l'] = abs(del_d_l) / df['isi']

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

    df.loc[:, 'del_d_r'] = del_d_r
    df.loc[:, 'del_d_l'] = del_d_l

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
    # print(df.head())

    # sample rate: 100 samples per sec
    df = df[1000:7000]

    # Remove NA's
    df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna()
    df.reset_index(drop=True, inplace=True)
    len_na = len_init - len(df)

    df.loc[:,'isi'] = np.diff(df.time, prepend=0)

    # Calculate amplitude, velocity, acceleration, change in acceleration
    df = calculate_features(df, smooth=False)


    ### STEP 1: Clean data based on eye physiology ###

    bad_data = [] # list of indices
    cond_1, cond_2, cond_3, cond_4, cond_5 = 0, 0, 0, 0, 0
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

    # Calculate target
    df.loc[:,'target_d'] = df['target_angle_x'] + df['target_angle_y']
    del_target_d = np.diff(df['target_d'], prepend=0)
    df.loc[:,'target_vel'] = abs(del_target_d) / df['isi']
    for i in range(1,len(df)-1):
        prev = df.target_vel[i-1]
        next = df.target_vel[i+1]
        if df.target_vel[i] == 0 and prev != 0 and next != 0:
            df.target_vel[i] = abs(next+prev)/2.0
    del_target_vel = np.diff(df.target_vel, prepend=0)
    df.loc[:,'target_accel'] = abs(del_target_vel) / df['isi']

    print("================= PREPROCESSING RESULTS =====================")
    print("len(df) before processing:", len_init)
    print("Number of inf and na removed:", len_na)
    print("\nNumber of datapoints with no pupil size:", cond_1)
    print("Negative intersample velocity:", cond_2)
    print("Intersample velocity greater than 1000 deg/s:", cond_3)

    print("\nlen of 'bad data':", len(bad_data))
    print("len of data after cleaning:", len(df))

    print("=============================================================")

    return df