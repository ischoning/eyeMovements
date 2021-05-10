# import sys
# sys.path.insert(1, "/Users/ischoning/PycharmProjects/eyeMovements/eventdetect-master")
import pandas as pd
from pylab import *
import matplotlib.pyplot as plt
import numpy as np
from Sample import Sample
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import IsolationForest
import common
import kmeans

def run_kmeans(df, eye = 'right'):
    if eye == 'right' or eye == 'Right':
        X = np.array([df.d_r, df.vel_vs_d_r]).T
    else:
        X = np.array([df.d_l, df.vel_vs_d_l]).T
    K = np.array([1, 2, 3, 4])
    costs = []
    for k in K:
        cost = []
        for i in range(0, 5):
            seed = np.random.seed(i)
            mixture, post = common.init(X, k, seed)
            kmixture, kpost, kcost = kmeans.run(X, mixture, post)
            common.plot(X, kmixture, kpost, "K = " + str(k) + ", seed = " + str(i) + ", Cost = " + str(kcost))
            cost.append(kcost)
        costs.append(min(cost))
    print("costs:",costs)

def find_best_fit(df, show_plot = False):
    # left eye
    X = np.array(df.d_l).reshape(-1, 1)
    y = np.array(df.vel_l).reshape(-1, 1)

    # identify outliers in the training dataset
    iso = IsolationForest(contamination=0.1)
    yhat = iso.fit_predict(X)
    # select all rows that are not outliers
    mask = yhat != -1
    X_train, y_train = X[mask, :], y[mask]
    # fit the model
    model = LinearRegression()
    model.fit(X_train, y_train)
    # evaluate the model
    y_left = model.predict(X)
    if show_plot:
        plt.scatter(df.d_l, df.vel_l, s = 2)
        plt.plot(X, y_left, color="green", label='best fit')
        plt.title('Left Eye: Amplitude vs Velocity')
        plt.xlabel('deg')
        plt.ylabel('deg/s')
        plt.legend()
        plt.show()

    # right eye
    X = np.array(df.d_r).reshape(-1, 1)
    y = np.array(df.vel_r).reshape(-1, 1)

    # identify outliers in the training dataset
    iso = IsolationForest(contamination=0.1)
    yhat = iso.fit_predict(X)
    # select all rows that are not outliers
    mask = yhat != -1
    X_train, y_train = X[mask, :], y[mask]
    # fit the model
    model = LinearRegression()
    model.fit(X_train, y_train)
    # evaluate the model
    y_right = model.predict(X)
    if show_plot:
        plt.scatter(df.d_r, df.vel_r, s = 2)
        plt.plot(df.d_r, y_right, color="green", label='best fit')
        plt.title('Right Eye: Amplitude vs Velocity')
        plt.xlabel('deg')
        plt.ylabel('deg/s')
        plt.legend()
        plt.show()

    return {'Left':y_left, 'Right':y_right}


def remove_outliers(df):
    """
    :param df: a dataframe with the below columns
    :param eye: left or right eye, defaults to left eye if no value provided
    :return: a dataframe cleaned from outliers
    """

    df = df[['time', 'dt', 'device_time', 'left_pupil_measure1', 'right_pupil_measure1', 'target_angle_x', 'target_angle_y', 'right_gaze_x',
             'right_gaze_y', 'left_angle_x', 'left_angle_y', 'right_angle_x', 'right_angle_y']]
    len_init = len(df)

    # sample rate: 100 samples per sec
    df = df[:1000]

    # Remove NA's
    df.replace([np.inf, -np.inf], np.nan)
    df.dropna(inplace=True)
    len_na = len_init - len(df)

    # Calculate amplitude, velocity, acceleration, change in acceleration
    df['isi'] = np.diff(df.time, prepend=0)

    df['d_r'] = np.sqrt(df.right_angle_x ** 2 + df.right_angle_y ** 2)
    df['d_l'] = np.sqrt(df.left_angle_x ** 2 + df.left_angle_y ** 2)

    del_d_r = np.diff(df.d_r, prepend=0)
    del_d_l = np.diff(df.d_l, prepend=0)

    df['vel_r'] = abs(del_d_r) / df.isi
    df['vel_l'] = abs(del_d_l) / df.isi

    del_vel_r = np.diff(df.vel_r, prepend=0)
    del_vel_l = np.diff(df.vel_l, prepend=0)

    df['accel_r'] = abs(del_vel_r) / df.isi
    df['accel_l'] = abs(del_vel_l) / df.isi

    del_accel_r = np.diff(df.accel_r, prepend=0)
    del_accel_l = np.diff(df.accel_l, prepend=0)

    df['jolt_r'] = del_accel_r / df.isi
    df['jolt_l'] = del_accel_l / df.isi

    # remove the first three datapoints (due to intersample calculations)
    df = df[3:]
    df.reset_index(drop=True, inplace=True)

    # explore thresholding
    df['d_vs_vel_l'] = np.divide(df.d_l, df.vel_l)
    plt.scatter(df.vel_l, df.d_vs_vel_l, s = 2)
    plt.title('left eye: dist vs ratio dist/vel')
    plt.xlabel('deg/s')
    plt.ylabel('ratio (d/vel)')
    plt.show()
    df['d_vs_vel_r'] = np.divide(df.d_r, df.vel_r)
    plt.scatter(df.vel_r, df.d_vs_vel_r, s=2)
    plt.title('right eye: dist vs ratio dist/vel')
    plt.xlabel('deg/s')
    plt.ylabel('ratio (d/vel)')
    plt.show()

    df['vel_vs_d_l'] = np.divide(df.vel_l, df.d_l)
    plt.scatter(df.d_l, df.vel_vs_d_l, s = 2)
    plt.title('left eye: dist vs ratio vel/dist')
    plt.xlabel('deg')
    plt.ylabel('ratio (vel/d)')
    plt.show()
    df['vel_vs_d_r'] = np.divide(df.vel_r, df.d_r)
    plt.scatter(df.d_l, df.vel_vs_d_r, s = 2)
    plt.title('right eye: dist vs ratio vel/dist')
    plt.xlabel('deg')
    plt.ylabel('ratio (vel/d)')
    plt.show()

    # make plots and return best fit
    best_fit = find_best_fit(df, show_plot = True)

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

        # angular velocity greater than 1000 deg/s?
        elif current['Left']['vel'] > 100 or current['Right']['vel'] > 100:
            bad_data.append(i)
            cond_2 += 1

        # correlation between amplitude of movement and velocity
        # Are there sudden changes in velocity without change in position or vice versa?
        # elif abs(best_fit['Left'][i] - current['Left']['vel'])/best_fit['Left'][i] > delta or abs(best_fit['Right'][i] - current['Right']['vel'])/best_fit['Right'][i] > delta:
        #     bad_data.append(i)
        #     cond_3 += 1

    # correlation between amplitude of movement and velocity
    # Are there sudden changes in velocity without change in position or vice versa?
    prev_len = len(df)
    df.drop(df[df.d_vs_vel_l > 50].index, inplace = True)
    df.drop(df[df.d_vs_vel_r > 50].index, inplace = True)
    cond_3 = prev_len - len(df)

    df.drop(index=bad_data, inplace=True)
    df.reset_index(drop=True, inplace=True)


    print("================= PREPROCESSING RESULTS =====================")
    print("len(df) before processing:", len_init)
    print("Number of inf and na removed:", len_na)
    print("\nNumber of datapoints with no pupil size:", cond_1)
    print("Intersample velocity equal to zero or greater than 1000 deg/s:", cond_2)
    print("Outliers:", cond_3)

    print("\nlen of 'bad data':", len(bad_data))
    print("len of data after cleaning:", len(df))

    print("=============================================================")

    df['d_vs_vel_l'] = np.divide(df.d_l, df.vel_l)
    plt.scatter(df.vel_l, df.d_vs_vel_l, s=2)
    plt.title('left eye: dist vs ratio dist/vel')
    plt.xlabel('deg/s')
    plt.ylabel('ratio (d/vel)')
    plt.show()
    df['d_vs_vel_r'] = np.divide(df.d_r, df.vel_r)
    plt.scatter(df.vel_r, df.d_vs_vel_r, s=2)
    plt.title('right eye: dist vs ratio dist/vel')
    plt.xlabel('deg/s')
    plt.ylabel('ratio (d/vel)')
    plt.show()

    df['vel_vs_d_l'] = np.divide(df.vel_l, df.d_l)
    plt.scatter(df.d_l, df.vel_vs_d_l, s=2)
    plt.title('left eye: dist vs ratio vel/dist')
    plt.xlabel('deg')
    plt.ylabel('ratio (vel/d)')
    plt.show()
    df['vel_vs_d_r'] = np.divide(df.vel_r, df.d_r)
    plt.scatter(df.d_l, df.vel_vs_d_r, s=2)
    plt.title('right eye: dist vs ratio vel/dist')
    plt.xlabel('deg')
    plt.ylabel('ratio (vel/d)')
    plt.show()

    run_kmeans(df)

    return df