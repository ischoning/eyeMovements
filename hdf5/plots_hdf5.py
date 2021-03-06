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

def get_feats(df):
    x = df.gaze_x
    y = df.gaze_y
    d = df.d
    v = df.vel
    a = df.accel

    return x, y, d, v, a

def plot_path(df):
    """ Show scatter plot of x and y position and target. """
    plt.scatter(df.gaze_x, df.gaze_y, label='sample', s=0.5)
    # plt.scatter(df.target_angle_x, df.target_angle_y, color='red', label='target', s=0.5)
    plt.title('After Cleaning')
    plt.xlabel('x (deg)')
    plt.ylabel('y (deg)')
    plt.legend()
    plt.show()


def plot_vs_time(df, feat, label = ''):
    """
    :param df: dataframe
    :param feat: feature of the dataframe to plot (d, v, a)
    :param label: {'Amplitude', 'Velocity', 'Acceleration'}
    :return:
    """
    if label == '':
        raise Exception('no label provided for plot title')

    head = label + ' vs Time'
    if label == 'Amplitude':
        ylabel = 'deg'
        plt_label = 'intersample distance'
    elif label == 'Velocity':
        ylabel = 'deg/s'
        plt_label = 'intersample velocity'
    elif label == 'Acceleration':
        ylabel = 'deg/s*s'
        plt_label = 'intersample acceleration'
    else:
        raise Exception("label should be either 'Amplitude', 'Velocity', or 'Acceleration'")

    fig, ax = plt.subplots(figsize=pltsize)

    x, y, d, v, a = get_feats(df)

    # plot vs time
    if label == 'Amplitude':
        ax.plot(df.time, x, label='x', linewidth = 0.5)
        ax.plot(df.time, y, label='y', color='navy', linewidth = 0.5)
        ax.plot(df.time, feat, label=plt_label, color='green')
        # ax.plot(df.time, df.target_d, color='red', label='target', linewidth = 0.5)
    elif label == 'Velocity':
        ax.plot(df.time, d, label = 'position', color = 'green', linewidth = 0.5)
        ax.plot(df.time, feat, label=plt_label, color='orange')
        # ax.plot(df.time, df.target_vel, color='red', label='target', linewidth = 0.5)
    elif label == 'Acceleration':
        ax.plot(df.time, d, label='position', color='green', linewidth = 0.5)
        ax.plot(df.time, df.vel, label=plt_label, color='orange', linewidth = 0.5)
        ax.plot(df.time, feat, label=plt_label, color='purple')
        # ax.plot(df.time, df.target_accel, color='red', label='target', linewidth=0.5)
    ax.legend()
    ax.set_xlabel('time (s)')
    ax.set_ylabel(ylabel)
    ax.set_title(head)

    plt.show()


def plot_events(df):

    x, y, d, v, a = get_feats(df)

    fig, ax = plt.subplots(figsize=pltsize)

    ax.plot(df.time, x, color='red')

    # https://matplotlib.org/3.1.1/gallery/lines_bars_and_markers/fill_between_demo.html
    trans = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
    min_val = min(x)
    max_val = max(x)
    ax.fill_between(df.time, min_val, max_val, where=df.event == 'Fix',
                    facecolor='green', alpha=0.5, transform=trans, label='fixation')
    try:
        ax.fill_between(df.time, min_val, max_val, where=df.event == 'SmP',
                        facecolor='red', alpha=0.5, transform=trans, label='smooth pursuit')
        ax.fill_between(df.time, min_val, max_val, where=df.event == 'Sac',
                        facecolor='blue', alpha=0.5, transform=trans, label='saccade')
    except:
        ax.fill_between(df.time, min_val, max_val, where=df.event == 'Sac',
                        facecolor='blue', alpha=0.5, transform=trans, label='other')
    ax.legend()

    ax.set_title('Classification of Angular Displacement Over Time')
    ax.set_xlabel('time (ms)')
    ax.set_ylabel('deg')

    plt.show()


def plot_hist(df, title, x_axis, density=False):
    """ Plot histogram of data passed in params. """

    x, y, d, v, a = get_feats(df)

    result, bin_edges = np.histogram(v, int(len(v)/2), density=density)
    plt.hist(v, bins=int(len(v)/2), density=density, label = 'intersample velocity', alpha = 0.5)
    # plt.hist(df.target_vel[df.target_vel != 0], bins=int(len(v)/2), density=density, color='red', label='target', alpha = 0.5)
    plt.title(title)
    plt.ylabel('Count')
    plt.xlabel(x_axis)
    plt.show()

    return result, bin_edges


def plot_pmf(data, title='Velocity', x_axis='deg/s'):
    return plot_hist(data, title=title, x_axis=x_axis, density=True)