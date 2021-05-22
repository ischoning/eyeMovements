# import sys
# sys.path.insert(1, "/Users/ischoning/PycharmProjects/eyeMovements/eventdetect-master")
from pylab import *
import numpy as np
from Sample_window import Sample
from Archive import common, kmeans
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


def remove_outliers(df):
    """
    :param df: a dataframe with the below columns
    :return: a dataframe cleaned from outliers
    """

    len_init = len(df)
    print(df)

    # sample rate: 100 samples per sec
    df = df[1000:7000]

    # Remove NA's
    df.replace([np.inf, -np.inf], np.nan)
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    len_na = len_init - len(df)


    # Smooth using kernel windows of sizes 3, 5, 7
    window_size = 7
    data = (df.left_angle_x, df.left_angle_y, df.right_angle_x, df.right_angle_y)
    for i in range(len(df)-window_size):
        for elem in data:
            # if first and last elements in window have less than 0.5 deg difference
            if abs(elem[i] - elem[i+window_size-1]) < 0.5:
                m = (elem[i] + elem[i+window_size-1])/2.0
                # set all elements in between equal to their average
                for j in range(i+1, i+window_size-1):
                    elem[j] = m

    df.left_angle_x = signal.savgol_filter(df.loc[:, 'left_angle_x'], window_length=101, polyorder=3)
    df.left_angle_y = signal.savgol_filter(df.loc[:, 'left_angle_y'], window_length=101, polyorder=3)
    df.right_angle_x = signal.savgol_filter(df.loc[:, 'right_angle_x'], window_length=101, polyorder=3)
    df.right_angle_y = signal.savgol_filter(df.loc[:, 'right_angle_y'], window_length=101, polyorder=3)

    df['isi'] = np.diff(df.time, prepend=0)

    # Calculate target
    df['target_d'] = df['target_angle_x'] + df['target_angle_y']
    del_target_d = np.diff(df['target_d'], prepend=0)
    df['target_vel'] = abs(del_target_d) / df['isi']
    for i in range(1,len(df)-1):
        prev = df.target_vel[i-1]
        next = df.target_vel[i+1]
        if df.target_vel[i] == 0 and prev != 0 and next != 0:
            df.target_vel[i] = abs(next+prev)/2.0
    del_target_vel = np.diff(df.target_vel, prepend=0)
    df['target_accel'] = abs(del_target_vel) / df['isi']

    # Calculate amplitude, velocity, acceleration, change in acceleration

    del_x_r = np.diff(df.right_angle_x, prepend=0)
    del_y_r = np.diff(df.right_angle_y, prepend=0)
    #df['d_r'] = np.sqrt(del_x_r ** 2 + del_y_r ** 2)
    df['d_r'] = df.right_angle_x + df.right_angle_y
    #df['d_r'] = np.convolve(df.right_angle_x, df.right_angle_y, mode='same')

    del_x_l = np.diff(df.left_angle_x, prepend=0)
    del_y_l = np.diff(df.left_angle_y, prepend=0)
    #df['d_l'] = np.sqrt(del_x_l ** 2 + del_y_l ** 2)
    df['d_l']  = df.left_angle_x + df.left_angle_y
    #df['d_l'] = np.convolve(df.left_angle_x, df.left_angle_y, mode='same')

    df['d_r'] = signal.savgol_filter(df.loc[:,'d_r'], window_length = 51, polyorder = 3)
    df['d_l'] = signal.savgol_filter(df.loc[:,'d_l'], window_length = 51, polyorder = 3)

    #scipy filtering (butterworth, wells..?)

    del_d_r = np.diff(df['d_r'], prepend=0)
    del_d_l = np.diff(df['d_l'], prepend=0)

    df['vel_r'] = abs(del_d_r) / df['isi']
    df['vel_l'] = abs(del_d_l) / df['isi']

    df['vel_r'] = signal.savgol_filter(df.loc[:,'vel_r'], window_length = 101, polyorder = 3)
    df['vel_l'] = signal.savgol_filter(df.loc[:,'vel_l'], window_length = 101, polyorder = 3)

    del_vel_r = np.diff(df['vel_r'], prepend=0)
    del_vel_l = np.diff(df['vel_l'], prepend=0)

    df['accel_r'] = abs(del_vel_r) / df['isi']
    df['accel_l'] = abs(del_vel_l) / df['isi']

    df['accel_r'] = signal.savgol_filter(df.loc[:,'accel_r'], window_length=101, polyorder=3)
    df['accel_l'] = signal.savgol_filter(df.loc[:,'accel_l'], window_length=101, polyorder=3)

    del_accel_r = np.diff(df['accel_r'], prepend=0)
    del_accel_l = np.diff(df['accel_l'], prepend=0)

    df['jolt_r'] = del_accel_r / df['isi']
    df['jolt_l'] = del_accel_l / df['isi']

    df['del_d_r'] = del_d_r
    df['del_d_l'] = del_d_l

    # remove the first three datapoints (due to intersample calculations)
    df = df[3:]
    df.reset_index(drop=True, inplace=True)


    # initialize class
    sample = Sample(df)
    # Clean data by eye physiology
    bad_data = []
    cond_1, cond_2, cond_3 = 0, 0, 0
    for i in range(1, len(df)-2):

        prev, current, next = sample.get_window(i)
        delta = 0.05

        # no pupil size?
        if current['Left']['pupil'] == 0 or current['Right']['pupil'] == 0:
            bad_data.append(i)
            cond_1 += 1

        # remove negative velocity (should be abs)
        elif current['Left']['vel'] < 0 or current['Right']['vel'] < 0:
            bad_data.append(i)
            cond_2 += 1

        # angular velocity greater than 1000 deg/s?
        elif current['Left']['vel'] > 1000 or current['Right']['vel'] > 1000:
            bad_data.append(i)
            cond_3 += 1

        # angular velocity greater than 1000 deg/s?
        # elif current['Left']['vel'] > 100 or current['Right']['vel'] > 100:
        #     bad_data.append(i)
        #     cond_2 += 1

        # correlation between amplitude of movement and velocity
        # Are there sudden changes in velocity without change in position or vice versa?
        # elif abs(best_fit['Left'][i] - current['Left']['vel'])/best_fit['Left'][i] > delta or abs(best_fit['Right'][i] - current['Right']['vel'])/best_fit['Right'][i] > delta:
        #     bad_data.append(i)
        #     cond_3 += 1

    # correlation between amplitude of movement and velocity
    # Are there sudden changes in velocity without change in position or vice versa?
    # prev_len = len(df)
    # df.drop(df[df.d_vs_vel_l > 50].index, inplace = True)
    # df.drop(df[df.d_vs_vel_r > 50].index, inplace = True)
    # cond_3 = prev_len - len(df)
    #num_removed = len(bad_data) + cond_3

    df.drop(index=bad_data, inplace=True)
    df.reset_index(drop=True, inplace=True)


    print("================= PREPROCESSING RESULTS =====================")
    print("len(df) before processing:", len_init)
    print("Number of inf and na removed:", len_na)
    print("\nNumber of datapoints with no pupil size:", cond_1)
    print("Negative intersample velocity:", cond_2)
    print("Intersample velocity greater than 1000 deg/s:", cond_3)
    #print("Intersample velocity equal to zero or greater than 1000 deg/s:", cond_2)
    #print("Outliers:", cond_3)

    print("\nlen of 'bad data':", len(bad_data))
    print("len of data after cleaning:", len(df))

    print("=============================================================")

    # df['d_vs_vel_l'] = np.divide(df.d_l, df.vel_l)
    # plt.scatter(df.vel_l, df.d_vs_vel_l, s=2)
    # plt.title('left eye: dist vs ratio dist/vel')
    # plt.xlabel('deg/s')
    # plt.ylabel('ratio (d/vel)')
    # plt.show()
    # df['d_vs_vel_r'] = np.divide(df.d_r, df.vel_r)
    # plt.scatter(df.vel_r, df.d_vs_vel_r, s=2)
    # plt.title('right eye: dist vs ratio dist/vel')
    # plt.xlabel('deg/s')
    # plt.ylabel('ratio (d/vel)')
    # plt.show()
    #
    # df['vel_vs_d_l'] = np.divide(df.vel_l, df.d_l)
    # plt.scatter(df.d_l, df.vel_vs_d_l, s=2)
    # plt.title('left eye: dist vs ratio vel/dist')
    # plt.xlabel('deg')
    # plt.ylabel('ratio (vel/d)')
    # plt.show()
    # df['vel_vs_d_r'] = np.divide(df.vel_r, df.d_r)
    # plt.scatter(df.d_l, df.vel_vs_d_r, s=2)
    # plt.title('right eye: dist vs ratio vel/dist')
    # plt.xlabel('deg')
    # plt.ylabel('ratio (vel/d)')
    # plt.show()

    #run_kmeans(df)

    return df