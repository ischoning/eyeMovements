import pandas as pd
from pylab import *
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import math
import matplotlib.transforms as mtransforms
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import IsolationForest


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


def plot_vs_time(df, eye = 'left'):
    """ Plots angular position vs time. """
    if eye == 'right' or eye == 'Right':
        x = df.right_angle_x
        y = df.right_angle_y
        d = df.d_r
        head = 'Right Eye: Position vs Time'
    else:
        x = df.left_angle_x
        y = df.left_angle_y
        d = df.d_l
        head = 'Left Eye: Position vs Time'

    # plot vs time
    plt.plot(df.time, x, label='x')
    plt.plot(df.time, y, label='y', color='red')
    plt.plot(df.time, d, label='intersample distance traveled', color='green')
    plt.legend()
    plt.xlabel('time')
    plt.ylabel('deg')
    plt.title(head)
    plt.show()


def plot_events(df, eye = 'Left'):

    if eye == 'Right' or eye == 'right':
        x = df.d_r
    else: x = df.d_l

    fig, ax = plt.subplots()

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

    ax.set_title(eye+'eye: '+'Classification of Angular Displacement Over Time')
    ax.set_xlabel('time (ms)')
    ax.set_ylabel('deg')

    plt.show()


def plot_hist(data, title, x_axis, density=False):
    """ Plot histogram of data passed in params. """
    if density:
        result, bin_edges = np.histogram(data, bins=len(data), density=True)
        plt.hist(data, bins=len(data), density=True)
    else:
        plt.hist(data, bins=int(len(data) / 2))
    plt.title(title)
    plt.ylabel('Count')
    plt.xlabel(x_axis)
    plt.show()

    if density:
        return result, bin_edges
    else:
        return None

def pmf(data, title='Velocity', x_axis='deg/s'):
    return plot_hist(data, title=title, x_axis=x_axis, density=True)