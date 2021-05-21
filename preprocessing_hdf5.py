# import sys
# sys.path.insert(1, "/Users/ischoning/PycharmProjects/eyeMovements/eventdetect-master")
import pandas as pd
from pylab import *
import matplotlib.pyplot as plt
import numpy as np
from Sample_hdf5 import Sample
import common
import kmeans
from scipy import signal


def remove_outliers(df, process = True):
    """
    :param df: a dataframe with the below columns
    :param hdf5: boolean (True if file is an hdf5, False otherwise)
    :return: a dataframe cleaned from outliers
    """

    df = df[['time', 'status', 'gaze_x', 'gaze_y', 'gaze_z', 'pupil_measure1']]

    # sample rate: 100 samples per sec
    df = df[:1000]
    len_init = len(df)

    # Remove NA's
    df.replace([np.inf, -np.inf], np.nan)
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    len_na = len_init - len(df)

    # Calculate amplitude, velocity, acceleration, change in acceleration

    # assign relevant data
    x = df['gaze_x']
    y = df['gaze_y']
    z = df['gaze_z']

    # compute angular values
    df['x'] = np.rad2deg(np.arctan2(x, z))
    df['y'] = np.rad2deg(np.arctan2(y, z))

    df['isi'] = np.diff(df.time, prepend=0)

    df['d'] = df['x'] + df['y']

    del_d = np.diff(df['d'], prepend=0)

    df['vel'] = abs(del_d) / df.isi

    del_vel = np.diff(df['vel'], prepend=0)

    df['accel'] = abs(del_vel) / df.isi

    del_accel = np.diff(df['accel'], prepend=0)

    df['jolt'] = del_accel / df.isi

    # remove the first three datapoints (due to intersample calculations)
    df = df[3:]
    df.reset_index(drop=True, inplace=True)


    if process:
        # initialize class
        sample = Sample(df)
        # Clean data by eye physiology
        bad_data = []
        cond_1, cond_2, cond_3 = 0, 0, 0
        for i in range(1, len(df)-2):

            prev, current, next = sample.get_window(i)
            delta = 0.05

            # no pupil size?
            if current['pupil'] == 0:
                bad_data.append(i)
                cond_1 += 1

            # remove negative velocity (should be abs)
            elif current['vel'] < 0:
                bad_data.append(i)
                cond_2 += 1

            # angular velocity greater than 1000 deg/s?
            elif current['vel'] > 1000:
                bad_data.append(i)
                cond_3 += 1


        df.drop(index=bad_data, inplace=True)
        df.reset_index(drop=True, inplace=True)


        print("================= PREPROCESSING RESULTS =====================")
        print("len(df) before processing:", len_init)
        print("Number of inf and na removed:", len_na)
        print("\nNumber of datapoints with no pupil size:", cond_1)
        print("Negative intersample velocity:", cond_2)
        print("Intersample velocity greater than 1000 deg/s:", cond_3)

        print("\nlen of 'bad data':", len(bad_data))
        print("len of data after cleaning:", len(df))

        print("=============================================================")

    else:
        print("================= PREPROCESSING RESULTS =====================")
        print("len(df) before processing:", len_init)
        print("Number of inf and na removed:", len_na)
        print("=============================================================")

    return df