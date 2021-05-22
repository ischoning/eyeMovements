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

    plt.scatter(df.right_angle_x, df.right_angle_y, label='sample right', s=0.5)
    plt.scatter(df.left_angle_x, df.left_angle_y, label='sample left', s=0.5, color='green')
    plt.scatter(df.target_angle_x, df.target_angle_y, color='red', label='target', s=0.5)
    plt.title('After Cleaning')
    plt.xlabel('x (deg)')
    plt.ylabel('y (deg)')
    plt.legend()
    plt.show()


def plot_vs_time(df, eye = 'left'):

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

def make_hist(data, title, x_axis, y_axis, density=False):
    if density:
        result, bin_edges = np.histogram(data, bins=len(data), density=True)
        plt.hist(data, bins=len(data), density=True)
    else:
        plt.hist(data, bins=int(len(data) / 2))
    plt.title(title)
    plt.ylabel(y_axis)
    plt.xlabel(x_axis)
    plt.show()

    if density:
        return result, bin_edges
    else:
        return None

def pmf(data, title, x_axis, y_axis):
    hist, bin_edges = make_hist(data, title, x_axis, y_axis, density=True)
    # print("hist sum:", np.sum(hist * np.diff(bin_edges))) # sanity check - should = 1

    return hist, bin_edges

def overview_plot(data):
    fig, ax = plt.subplots()
    fig.subplots_adjust(right=0.75)

    ax1 = ax.twinx()
    ax2 = ax.twinx()

    # Offset the right spine of twin2.  The ticks and label have already been
    # placed on the right by twinx above.
    # ax2.spines.right.set_position(("axes", 1.2))

    p1, = ax.plot(data.sample_i, data.position, "b-", label="Displacement")
    p2, = ax1.plot(data.sample_i, data.velocity, "r-", label="Velocity")
    p3, = ax2.plot(data.sample_i, data.acceleration, "g-", label="Acceleration")

    # ax.set_xlim(0, 2)
    # ax.set_ylim(0, 2)
    # ax1.set_ylim(0, 4)
    # ax2.set_ylim(1, 65)

    ax.set_xlabel("sample sequential number")
    ax.set_ylabel("Degree")
    ax1.set_ylabel("Deg/Ms")
    ax2.set_ylabel("Deg/Ms^2")

    ax.yaxis.label.set_color(p1.get_color())
    ax1.yaxis.label.set_color(p2.get_color())
    ax2.yaxis.label.set_color(p3.get_color())

    tkw = dict(size=4, width=1.5)
    ax.tick_params(axis='y', colors=p1.get_color(), **tkw)
    ax1.tick_params(axis='y', colors=p2.get_color(), **tkw)
    ax2.tick_params(axis='y', colors=p3.get_color(), **tkw)
    ax.tick_params(axis='x', **tkw)

    ax.legend(handles=[p1, p2, p3])

    plt.show()

def plot(df):
    # left eye

    plt.scatter(df.d_l, df.vel_l, s=0.5)
    # A = np.linspace(np.min(df.d_l), np.max(df.d_l), 100)
    # D = 21+2.2*A
    # plt.plot(A, D, ':', color = 'orange', label = 'Carpenters Thm: D = 21 + 2.2*A')
    plt.scatter(df.vel_l, df.accel_l, s=0.5)
    plt.title('Left Eye: Velocity vs Acceleration')
    plt.xlabel('deg/s')
    plt.ylabel('deg/s^2')
    plt.ylim((0, 7500))
    plt.xlim((0, 400))
    plt.show()
    plt.scatter(df.d_l, df.accel_l, s=0.5)
    plt.title('Left Eye: Amplitude vs Acceleration')
    plt.xlabel('deg')
    plt.ylabel('deg/s^2')
    plt.ylim((0, 7500))
    plt.show()

    # 3d plot
    # fig = plt.figure(figsize=(10, 7))
    # ax = plt.axes(projection="3d")
    # ax.scatter3D(df.d_l, df.vel_l, df.accel_l, s = 0.5, color="green")
    # plt.xlabel('Amp')
    # plt.ylabel('Vel')
    # plt.title("Left: Amp vs Vel vs Accel")
    # plt.show()

    # right eye

    plt.scatter(df.d_r, df.vel_r, s=0.5)
    # A = np.linspace(np.min(df.d_r), np.max(df.d_r), 100)
    # D = 21+2.2*A
    # plt.plot(A, D, ':', color = 'orange', label = 'Carpenters Thm: D = 21 + 2.2*A')
    plt.scatter(df.vel_r, df.accel_r, s=0.5)
    plt.title('Right Eye: Velocity vs Acceleration')
    plt.xlabel('deg/s')
    plt.ylabel('deg/s^2')
    plt.ylim((0, 7500))
    plt.xlim((0, 400))
    plt.show()
    plt.scatter(df.d_r, df.accel_r, s=0.5)
    plt.title('Right Eye: Amplitude vs Acceleration')
    plt.xlabel('deg')
    plt.ylabel('deg/s^2')
    plt.ylim((0, 7500))
    plt.show()

    # 3d plot
    # fig = plt.figure(figsize=(10, 7))
    # ax = plt.axes(projection="3d")
    # ax.scatter3D(df.d_r, df.vel_r, df.accel_r, s = 0.5, color="green")
    # plt.title("Right: Amp vs Vel vs Accel")
    # plt.xlabel('Amp')
    # plt.ylabel('Vel')
    # plt.show()