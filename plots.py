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
    e = Circle( xy=xy, radius=radius )
    if ax is None:
        ax = plt.gca()  # ax = subplot( 1,1,1 )
    ax.add_artist(e)
    e.set_clip_box(ax.bbox)
    e.set_edgecolor( color )
    e.set_facecolor( facecolor )  # "none" not None
    e.set_alpha( alpha )

def plot_path(df):
    """ Show scatter plot of x and y position and target. """
    plt.scatter(df['right_angle_x'], df['right_angle_y'], label='sample right', s=0.5)
    plt.scatter(df['left_angle_x'], df['left_angle_y'], label='sample left', s=0.5, color='green')
    try:
        plt.scatter(df['target_angle_x'], df['target_angle_y'], color='red', label='target', s=0.5)
    except: pass
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

    _, _, x, y, d, v, a, del_d = get_feats(df, eye)

    # plot vs time
    if label == 'Amplitude':
        ax.plot(df['time'], x, label='x', linewidth = 0.5)
        ax.plot(df['time'], y, label='y', color='navy', linewidth = 0.5)
        ax.plot(df['time'], feat, label=plt_label, color='green')
        try:
            ax.plot(df['time'], df['target_d'], color='red', label='target', linewidth = 0.5)
        except: pass
    elif label == 'Velocity':
        ax.plot(df['time'], d, label = 'position', color = 'green', linewidth = 0.5)
        ax.plot(df['time'], feat, label=plt_label, color='orange')
        try:
            ax.plot(df['time'], df['target_vel'], color='red', label='target', linewidth = 0.5)
        except: pass
    elif label == 'Acceleration':
        ax.plot(df['time'], d, label='position', color='green', linewidth = 0.5)
        ax.plot(df['time'], df.v, label=plt_label, color='orange', linewidth = 0.5)
        ax.plot(df['time'], feat, label=plt_label, color='purple')
        try:
            ax.plot(df['time'], df['target_accel'], color='red', label='target', linewidth=0.5)
        except: pass
    ax.legend()
    ax.set_xlabel('time (s)')
    ax.set_ylabel(ylabel)
    ax.set_title(head)

    plt.show()


def plot_events(df, eye = 'Left', method = ''):

    _, _, x, y, d, v, a, del_d = get_feats(df, eye)

    fig, ax = plt.subplots(figsize=pltsize)

    ax.plot(df['time'], d, color='red')

    # https://matplotlib.org/3.1.1/gallery/lines_bars_and_markers/fill_between_demo.html
    trans = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
    min_val = min(d)
    max_val = max(d)
    ax.fill_between(df['time'], min_val, max_val, where=df.event == 'fix',
                    facecolor='green', alpha=0.5, transform=trans, label='fixation')
    # ax.fill_between(df['time'], min_val, max_val, where=df.event == 'other',
    #                 facecolor='turquoise', alpha=0.5, transform=trans, label='other')
    try:
        ax.fill_between(df['time'], min_val, max_val, where=df.event == 'smp',
                        facecolor='orange', alpha=0.5, transform=trans, label='smooth pursuit')
    except: pass
    try:
        ax.fill_between(df['time'], min_val, max_val, where=df.event == 'sac',
                        facecolor='blue', alpha=0.5, transform=trans, label='saccade')
    except: pass

    ax.legend()
    ax.set_title('['+method+'] '+eye+' eye: '+'Classification of Angular Displacement Over Time')
    ax.set_xlabel('time (ms)')
    ax.set_ylabel('deg')

    plt.show()


def plot_hist(df, method, eye, title, x_axis, density=False):
    """ Plot histogram of data passed in params. """

    _, _, x, y, d, v, a, del_d = get_feats(df, eye)

    if method == 'Velocity':
        result, bin_edges = np.histogram(v, int(len(v)/2), density=density)
        plt.hist(v, bins=int(len(v)/2), density=density, label = 'intersample velocity', alpha = 0.5)
        try:
            plt.hist(df.target_vel[df['target_vel'] != 0], bins=int(len(v)/2), density=density, color='red', label='target', alpha = 0.5)
        except: pass
    elif method == 'Dispersion':
        result, bin_edges = np.histogram(np.abs(del_d), int(len(del_d) / 2), density=density)
        plt.hist(np.abs(d), bins=int(len(del_d) / 2), density=density, label='intersample displacement', alpha=0.5)
    plt.title(title)
    plt.ylabel('Count')
    plt.xlabel(x_axis)
    plt.legend()
    plt.show()

    return result, bin_edges


def plot_pmf(data, title='Velocity', x_axis='deg/s'):
    return plot_hist(data, title=title, x_axis=x_axis, density=True)



def plot_fixations_IDT(df,window_sizes, threshes):

    fig, ax = plt.subplots(len(threshes),len(window_sizes), figsize=(16,10))

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
                #print(f)
            #print("Number of fix events:", len(centers))
            #print("Number of fix samples:", np.sum(num_samples))

            # label the fixations in the dataframe
            df['event'] = 'other'
            count = 0
            #print('len(centers):', len(centers))
            for i in range(len(starts)):
                df.loc[starts[i]:ends[i], ("event")] = 'fix'
                # if the end of the data is all fixations
                if i == len(starts)-1:
                    df.loc[starts[i]:len(starts), ("event")] = 'fix'
                # if there are only 1 or 2 samples between fixations, combine them
                elif starts[i+1]-ends[i] <= 2:
                    count += 1
                    df.loc[ends[i]:starts[i+1], ("event")] = 'fix'
            #print(count)

            # # label the saccades using velocity threshold (22 deg/s according to Houpt)
            # df['event'] = np.where(df.v >= 22, 'sac', df.event)

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




def plot_fixations_IVT(df,threshes):


    fig, ax = plt.subplots(1,len(threshes), figsize=(16,6))

    ncol = 0
    for thresh in threshes:
        samples = [Sample(ind=i, time=df.time[i], x=df.x[i], y=df.y[i]) for i in range(len(df))]
        stream = ListSampleStream(samples)
        fixes = Velocity(sampleStream = IntersampleVelocity(stream), threshold = thresh)
        centers = []
        num_samples = []
        starts = []
        ends = []
        for f in fixes:
            centers.append(f.get_center())
            num_samples.append(f.get_num_samples())
            starts.append(f.get_start())
            ends.append(f.get_end())
            #print(f)
        #print("Number of fix events:", len(centers))
        #print("Number of fix samples:", np.sum(num_samples))

        # label the fixations in the dataframe
        df['event'] = 'other'
        count = 0
        #print('len(centers):', len(centers))
        for i in range(len(starts)):
            df.loc[starts[i]:ends[i], ("event")] = 'fix'
            # if the end of the data is all fixations
            if i == len(starts)-1:
                df.loc[starts[i]:len(starts), ("event")] = 'fix'
            # if there are only 1 or 2 samples between fixations, combine them
            elif starts[i+1]-ends[i] <= 2:
                count += 1
                df.loc[ends[i]:starts[i+1], ("event")] = 'fix'
        #print(count)

        # # label the saccades using velocity threshold (22 deg/s according to Houpt)
        # df['event'] = np.where(df.v >= 22, 'sac', df.event)

        centers = np.array(centers)
        ax[ncol].scatter(df.x[df.event !='fix'], df.y[df.event!='fix'], s=0.5,label='other')
        ax[ncol].scatter(df.x[df.event =='fix'], df.y[df.event =='fix'], color='r', s=0.5, label='fix')
        # ax[ncol].scatter(df.x[df.event == 'sac'], df.y[df.event == 'sac'], color='orange', s=0.5, label='sac')
        # for i in range(len(centers)):
        #     plots.circle(centers[i], radius=num_samples[i]*0.5+10)
        #plt.scatter(centers[:,0], centers[:,1], c='None', edgecolors='r')
        ax[ncol].set_title('[I-VT] Thresh: '+str(thresh))
        ax[ncol].set_xlabel('x pixel')
        ax[ncol].set_ylabel('y pixel')
        ax[ncol].legend()

        ncol += 1
    plt.legend()
    plt.show()