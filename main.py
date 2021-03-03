# import sys
# sys.path.insert(1, "/Users/ischoning/PycharmProjects/eyeMovements/eventdetect-master")
import pandas as pd
from pylab import *
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import math


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

def plot_vs_time(t, x, y = [], title = None, y_axis = None):
    '''
    input: two values (x,y) to be plotted against time
    output: two plots
    return: None
    '''
    if len(y) == 0:
        plt.plot(t, x, 'r')
    else:
        plt.plot(t, x, 'r', label='x')
        plt.plot(t, y, 'b', label='y')
        plt.legend()
    plt.title(title)
    plt.xlabel('timestamp')
    plt.ylabel(y_axis)

    plt.show()

    return None


def make_hist(data, title, x_axis):
    plt.hist(data, bins=100)
    plt.title(title)
    plt.ylabel('number of occurrences')
    plt.xlabel(x_axis)
    plt.show()


def remove_outliers(data):
    ''' source: https://www.statology.org/remove-outliers-python/ '''

    # ----Method 1---- z-score method
    #
    # find absolute value of z-score for each observation
    z = np.abs(stats.zscore(data))
    # only keep rows in dataframe with all z-scores less than absolute value of 3
    data_clean = data[(z < 3).all(axis=1)]

    # ----Method 2---- interquartile range method
    #
    # #find Q1, Q3, and interquartile range for each column
    # Q1 = data.quantile(q=.25)
    # Q3 = data.quantile(q=.75)
    # IQR = data.apply(stats.iqr)
    #
    # #only keep rows in dataframe that have values within 1.5*IQR of Q1 and Q3
    # data_clean = data[~((data < (Q1-1.5*IQR)) | (data > (Q3+1.5*IQR))).any(axis=1)]
    #

    # find how many rows are left in the dataframe
    print(data_clean.shape)
    return data_clean


def main():

    # files
    df = pd.read_csv('participant07_preprocessed172.csv')

    # shorten dataset for better visuals and quicker results
    #df = df[int(len(df)/200):int(len(df)/100)]
    df = df[0:int(len(df)/100)]

    # assign relevant data
    lx = df['left_forward_x']
    ly = df['left_forward_y']
    lz = df['left_forward_z']
    rx = df['right_forward_x']
    ry = df['right_forward_y']
    rz = df['right_forward_z']
    t = df['raw_timestamp']

    # compute angular values
    df['Ax_left'] = np.rad2deg(np.arctan2(lx, lz))
    df['Ay_left'] = np.rad2deg(np.arctan2(ly, lz))
    df['Ax_right'] = np.rad2deg(np.arctan2(rx, rz))
    df['Ay_right'] = np.rad2deg(np.arctan2(ry, rz))
    df['Avg_angular_x'] = df[['Ax_left', 'Ax_right']].mean(axis=1)
    df['Avg_angular_y'] = df[['Ay_left', 'Ay_right']].mean(axis=1)

    # show vision path in averaged angular degrees
    show_path(df['Avg_angular_x'], df['Avg_angular_y'])
    print('Length of capture time:', len(t))
    print('Length of capture time differences:',
          len(np.diff(t/1000000)))

    # # show vision path, separately for each eye
    plot_eye_path(df)

    # show angular displacement over time, averaged over both eyes
    plot_vs_time(t, df['Avg_angular_x'], df['Avg_angular_y'], 'Angular Displacement Over Time', 'degrees')

    # plot angular velocity for x and y
    dt = np.diff(t) # aka isi
    dx = np.diff(df['Avg_angular_x'])
    dy = np.diff(df['Avg_angular_y'])
    plot_vs_time(t[:len(df)-1],dx/dt,dy/dt, 'Angular Velocity Over Time', 'degrees per second')

    # plot combined angular velocity
    dr = np.sqrt(np.square(dx) + np.square(dy))
    plot_vs_time(t[:len(df)-1], dr, y = [], title = 'Combined Angular Velocity Over Time', y_axis = 'degrees per second')

    # show histogram of angular velocity
    make_hist(df, 'Histogram of Angular Velocity', 'angular velocity')

    #

"""
# ------ X --------
    # remove nans
    dx_dt = dx_dt[np.logical_not(np.isnan(dx_dt))]
    print(np.isnan(dx_dt).sum())
    plt.hist(dx_dt,50)
    plt.xlabel('dx/dt: angular velocity in x')
    plt.ylabel('number of occurrences')
    plt.show()

    # remove outliers
    z = np.abs(stats.zscore(dx_dt))
    # only keep rows in dataframe with all z-scores less than absolute value of 3
    dx_dt = dx_dt[(z < 3)]
    # plot histogram of angular velocities
    plt.hist(dx_dt,50)
    plt.xlabel('dx/dt: angular velocity in x')
    plt.ylabel('number of occurrences')
    plt.show()

# ------ Y --------
    # remove nans
    dy_dt = dy_dt[np.logical_not(np.isnan(dy_dt))]
    print(np.isnan(dy_dt).sum())
    plt.hist(dy_dt,50)
    plt.xlabel('dy/dt: angular velocity in y')
    plt.ylabel('number of occurrences')
    plt.show()

    # remove outliers
    z = np.abs(stats.zscore(dy_dt))
    # only keep rows in dataframe with all z-scores less than absolute value of 3
    dy_dt = dy_dt[(z < 3)]
    # plot histogram of angular velocities
    plt.hist(dy_dt,50)
    plt.xlabel('dy/dt: angular velocity in y')
    plt.ylabel('number of occurrences')
    plt.show()

    # segment graph given a condition (display on plot and return locs)
    #df_no_noise = disp(df)

    # smooth x and y movement using data from both eye and probability theory,
    # ie minimize noise (EyeLink algorithm)

    # next step: use LMS filter and HMM filter to identify fixation and saccade
    # compare results. MarkEye?
"""

if __name__ == "__main__":
    # Testing
    # hello("Isabella")
    main()