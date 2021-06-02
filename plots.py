import pandas as pd
from pylab import *
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import math
import matplotlib.transforms as mtransforms
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import IsolationForest
from preprocessing import get_feats
from matplotlib.patches import Circle  # $matplotlib/patches.py
from detect.dispersion import Dispersion
from detect.velocity import Velocity
from detect.intersamplevelocity import IntersampleVelocity
from detect.sample import Sample
from detect.sample import ListSampleStream


global pltsize
pltsize = (16,4)


def circle( xy, radius, color="red", facecolor="none", alpha=1, ax=None ):
    """ add a circle to ax= or current axes
    """
        # from .../pylab_examples/ellipse_demo.py
    e = Circle(xy=xy, radius=radius)
    if ax is None:
        ax = plt.gca()  # ax = subplot( 1,1,1 )
    ax.add_artist(e)
    e.set_clip_box(ax.bbox)
    e.set_edgecolor(color)
    e.set_facecolor(facecolor)  # "none" not None
    e.set_alpha(alpha)

def plot_path(df):
    """ Show scatter plot of x and y position and target. """
    if 'ppd_x' in df:
        plt.scatter(df['angle_x'], df['angle_y'], label='sample', s=0.5)
    else:
        plt.scatter(df['right_angle_x'], df['right_angle_y'], label='sample right', s=0.5)
        plt.scatter(df['left_angle_x'], df['left_angle_y'], label='sample left', s=0.5, color='green')
    if 'target_angle_x' in df.columns:
        plt.scatter(df['target_angle_x'], df['target_angle_y'], color='red', label='target', s=0.5)
    plt.title('After Cleaning')
    plt.xlabel('x (deg)')
    plt.ylabel('y (deg)')
    plt.legend()
    plt.show()


def plot_vs_time(df, label = '', eye = '', classify = False, method = '', show_target = False):
    """
    :param df: dataframe
    :param feat: feature of the dataframe to plot (d, v, a)
    :param label: {'Amplitude', 'Velocity', 'Acceleration'}
    :param eye:
    :return:
    """
    head = eye + ' Behavior over time'
    if label == 'Acceleration':
        ylabel = 'deg/s^2'
        plt_label = 'intersample acceleration'

    fig, ax = plt.subplots(figsize=pltsize)
    # twin object for two different y-axis on the sample plot
    ax2 = ax.twinx()

    # x, y, d, v, a, del_d = get_feats(df, eye)
    x, y, v, a = get_feats(df, eye)

    # plot vs time
    if label == 'Amplitude' or label == 'Velocity' or label == '':
        ax2.plot(df['time'], x, label='x', linewidth = 0.65)
        ax2.plot(df['time'], y, label='y', color='navy', linewidth = 0.65)
        ax2.set_ylabel('deg', color = 'navy', fontsize=14)
        # ax.plot(df['time'], feat, label=plt_label, color='green')
        # make a plot with different y-axis using second axis object
        ax.plot(df['time'], v, label='intersample velocity', color='red', linewidth = 0.9)
        ax.set_ylabel('deg/s', color='red', fontsize=14)
        if show_target and 'target_d' in df.columns:
            ax2.plot(df['time'], df['target_d'], color='black', label='target', linewidth = 0.35)
        if show_target and 'target_vel' in df.columns:
            ax.plot(df['time'], df['target_vel'], color='black', label='target', linewidth = 0.35)
        if classify:
            # https://matplotlib.org/3.1.1/gallery/lines_bars_and_markers/fill_between_demo.html
            trans = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
            min_val = min(v)
            max_val = max(v)
            ax.fill_between(df['time'], min_val, max_val, where=df.event == 'fix',
                            facecolor='green', alpha=0.35, transform=trans, label='fixation')
            # ax.fill_between(df['time'], min_val, max_val, where=df.event == 'other',
            #                 facecolor='turquoise', alpha=0.5, transform=trans, label='other')
            try:
                ax.fill_between(df['time'], min_val, max_val, where=df.event == 'smp',
                                facecolor='orange', alpha=0.35, transform=trans, label='smooth pursuit')
            except:
                pass
            try:
                ax.fill_between(df['time'], min_val, max_val, where=df.event == 'sac',
                                facecolor='blue', alpha=0.35, transform=trans, label='saccade')
            except:
                pass
            head = '[' + method + '] ' + eye + ' Event Classification'
    elif label == 'Acceleration':
        # ax.plot(df['time'], d, label='position', color='green', linewidth = 0.5)
        ax2.plot(df['time'], a, label='intersample acceleration', color='navy', linewidth=0.65)
        ax2.set_ylabel('deg/s^2', color='navy', fontsize=14)
        # make a plot with different y-axis using second axis object
        ax.plot(df['time'], v, label='intersample velocity', color='red', linewidth = 0.9)
        ax.set_ylabel('deg/s', color='red', fontsize=14)
        if show_target and 'target_accel' in df.columns:
            ax.plot(df['time'], df['target_accel'], color='black', label='target', linewidth=0.35)
    ax.legend(loc = 'upper left')
    ax2.legend(loc = 'upper right')
    ax.set_xlabel('time (s)')
    ax.set_title(head)

    plt.show()


def plot_vel_hist(df, eye, title, density=False, classify = False):
    """ Plot histogram of data passed in params. """

    x, y, v, a = get_feats(df, eye)
    num_bins = range(int(math.floor(np.min(df.v))),int(math.ceil(np.max(df.v))),1)

    # plot histogram
    if classify:
        df.loc[:,'v'] = np.round(df.v, 1)
        plt.hist(df[df.event == 'fix'].v, bins = num_bins, color = 'green', density = density, label = 'fixations', alpha = 0.5)
        plt.hist(df[df.event == 'smp'].v, bins = num_bins, color = 'orange', density = density, label = 'smooth pursuit', alpha = 0.75)
        plt.hist(df[df.event == 'sac'].v, bins = num_bins, color = 'blue', density = density, label = 'saccades', alpha = 0.5)
    else:
        v = np.round(v, 1)
        plt.hist(v, bins = num_bins, density = density, label = 'intersample velocity', alpha = 0.5)

    plt.title(title)
    if density:
        plt.ylabel('Percentage')
    else:
        plt.ylabel('Frequency')
    plt.xlabel('Intersample Velocity [deg/s]')
    plt.legend()
    plt.show()

    return


def plot_IDT_thresh_results(df, window_sizes, threshes):

    try:
        fig, ax = plt.subplots(len(threshes),len(window_sizes), figsize=(4*len(window_sizes),4*len(threshes)))
    except:
        raise Exception("window_sizes and threshes must be larger than length 1")

    nrow = 0
    ncol = 0
    for window_size in window_sizes:
        for thresh in threshes:
            samples = [Sample(ind=i, time=df.time[i], x=df.x[i], y=df.y[i]) for i in range(len(df))]
            stream = ListSampleStream(samples)
            fixes = Dispersion(sampleStream = stream, windowSize = window_size, threshold = thresh)
            centers = []
            num_samples = []
            starts = []
            ends = []
            for f in fixes:
                centers.append(f.get_center())
                num_samples.append(f.get_num_samples())
                starts.append(f.get_start())
                ends.append(f.get_end())

            # label the fixations in the dataframe
            df['event'] = 'other'
            count = 0
            for i in range(len(starts)):
                df.loc[starts[i]:ends[i], ("event")] = 'fix'
                # if the end of the data is all fixations
                if i == len(starts)-1:
                    df.loc[starts[i]:len(starts), ("event")] = 'fix'
                # if there are only 1 or 2 samples between fixations, combine them
                elif starts[i+1]-ends[i] <= 2:
                    count += 1
                    df.loc[ends[i]:starts[i+1], ("event")] = 'fix'

            centers = np.array(centers)
            ax[nrow][ncol].scatter(df.x[df.event !='fix'], df.y[df.event!='fix'], s=0.5,label='other')
            ax[nrow][ncol].scatter(df.x[df.event =='fix'], df.y[df.event =='fix'], color='r', s=0.5, label='fix')
            # ax[nrow][ncol].scatter(df.x[df.event =='sac'], df.y[df.event =='sac'], color='orange', s=0.5, label='sac')
            # for i in range(len(centers)):
            #     plots.circle(centers[i], radius=num_samples[i]*0.5+10)
            #plt.scatter(centers[:,0], centers[:,1], c='None', edgecolors='r')
            ax[nrow][ncol].set_title('[I-DT] Window: '+str(window_size)+' Thresh: '+str(thresh))
            ax[nrow][ncol].set_xlabel('x pixel')
            ax[nrow][ncol].set_ylabel('y pixel')
            ax[nrow][ncol].legend()

            nrow += 1
        nrow = 0
        ncol += 1
    plt.legend()
    plt.show()
