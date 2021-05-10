import pandas as pd
from pylab import *
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import math
import matplotlib.transforms as mtransforms
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import IsolationForest

global pltsize
pltsize = (16,4)

def plot_path(df):
    """ Show scatter plot of x and y position and target. """
    plt.scatter(df.right_angle_x, df.right_angle_y, label='sample right', s=0.5)
    plt.scatter(df.left_angle_x, df.left_angle_y, label='sample left', s=0.5, color='green')
    plt.scatter(df.target_angle_x, df.target_angle_y, color='red', label='target', s=0.5)
    plt.title('After Cleaning')
    plt.xlabel('x (deg)')
    plt.ylabel('y (deg)')
    plt.legend()
    plt.show()


def plot_vs_time(df, feat, label = '', eye = 'left'):
    """
    :param df: dataframe
    :param feat: feature of the dataframe to plot (d, v, a)
    :param label: {'Amplitude', 'Velocity', 'Acceleration'}
    :param eye:
    :return:
    """
    if label == '':
        raise Exception('no label provided for plot title')

    head = eye + ' eye: ' + label + ' vs Time'
    if label == 'Amplitude': ylabel = 'deg'
    elif label == 'Velocity': ylabel = 'deg/s'
    elif label == 'Acceleration': ylabel = 'deg/s*s'
    else: raise Exception("label should be either 'Amplitude', 'Velocity', or 'Acceleration'")

    fig, ax = plt.subplots(figsize=pltsize)

    if eye == 'right' or eye == 'Right':
        x = df.right_angle_x
        y = df.right_angle_y
    else:
        x = df.left_angle_x
        y = df.left_angle_y

    # plot vs time
    ax.plot(df.time, x, label='x')
    ax.plot(df.time, y, label='y', color='red')
    ax.plot(df.time, feat, label='intersample distance traveled', color='green')
    ax.legend()
    ax.set_xlabel('time (s)')
    ax.set_ylabel(ylabel)
    ax.set_title(head)

    plt.show()


def plot_events(df, eye = 'Left'):

    if eye == 'Right' or eye == 'right':
        x = df.d_r
    else: x = df.d_l

    fig, ax = plt.subplots(figsize=pltsize)

    ax.plot(df.time, x, color='red')

    # https://matplotlib.org/3.1.1/gallery/lines_bars_and_markers/fill_between_demo.html
    trans = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
    min_val = min(x)
    max_val = max(x)
    ax.fill_between(df.time, min_val, max_val, where=df.event == 'Fix',
                    facecolor='green', alpha=0.5, transform=trans, label='fixation')
    ax.fill_between(df.time, min_val, max_val, where=df.event == 'Sac',
                    facecolor='red', alpha=0.5, transform=trans, label='saccade')
    ax.legend()

    ax.set_title(eye+' eye: '+'Classification of Angular Displacement Over Time')
    ax.set_xlabel('time (ms)')
    ax.set_ylabel('deg')

    plt.show()


def plot_hist(data, title, x_axis, density=False):
    """ Plot histogram of data passed in params. """

    result, bin_edges = np.histogram(data, bins=len(data), density=density)
    plt.hist(data, bins=int(len(data)/2), density=density)
    plt.title(title)
    plt.ylabel('Count')
    plt.xlabel(x_axis)
    plt.show()

    return result, bin_edges


def pmf(data, title='Velocity', x_axis='deg/s'):
    return plot_hist(data, title=title, x_axis=x_axis, density=True)