# import sys
# sys.path.insert(1, "/Users/ischoning/PycharmProjects/eyeMovements/eventdetect-master")
import pandas as pd
from pylab import *
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
#import dispersion
#import eventstream

# from detect import *
# from detect.sample import Sample
# from detect.sample import FileSampleStream
#
# from detect.dispersion import *
# from detect.velocity import *
# from detect.hmm import *
# from detect.aoi import *
# from detect.movingaverage import *
# from detect.srr import *

def hello(name):
    print(f"Hello {name}!")

def make_df(file):
    '''
    input: a csv that includes 'Left(Right)EyeForward_x(y)' in header
    return: a pandas dataframe in floats
    '''
    # read in file
    data = pd.read_csv(file, sep=';', engine='python', decimal=',')
    df = data[['CaptureTime',
               'LeftEyeForward_x', 'LeftEyeForward_y', 'LeftEyeForward_z',
               'RightEyeForward_x', 'RightEyeForward_y', 'RightEyeForward_z']]

    # format to floats
    # replace scientific notation with 0
    for col in df:
        f = lambda x: x.apply(str).str.replace(".", "").str.contains('e', na=False, case=False)
        df1 = df.iloc[:, 0:].apply(f, axis=0)
        df.loc[(df1[col] == 1), col] = 0
        df[col] = df[col].astype(float)

    # check output
    #print(df.head())
    #print(df.dtypes)

    return df

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

def plot_vs_time(t, x, y):
    '''
    input: two values (x,y) to be plotted against time
    output: two plots
    return: None
    '''

    plt.plot(t, x, 'r', label='x')
    plt.plot(t, y, 'b', label='y')
    plt.title('Angular Displacement Over Time')
    plt.legend()
    plt.show()

    return None

def find_thresh(df):
    '''
    :return: histogram of saccade speeds
    '''
    dx = np.diff(df['Angular_x'])
    dy = np.diff(df['Angular_y'])
    dt = np.diff(df['CaptureTime'] / 1000000)

    for i in range(len(df)-1):
        speed = abs(dy[i]/dx[i])

    return thresh

def disp(df):
    '''
    Uses predefined thresholds for event categorization, ie Identification by
     dispersion threshold (I-DT).
    A simple function that categorizes eye movement as fixation or
    saccade by analyzing change in location.
    input: pandas dataframe that includes 'Left(Right)EyeForward_x(y)' in header, angular flag
    output: two plots classifying fixations and saccades for each eye
    return: None
    '''
    new_df = df.copy()

    human_thresh = 2.6

    dt = np.diff(df['CaptureTime'])
    dx_dt = np.diff(df['Avg_angular_x'])/dt
    dy_dt = np.diff(df['Avg_angular_y'])/dt
    plot_vs_time(df['CaptureTime'][:len(df)-1],dx_dt,dy_dt)

    saccades = []
    fixations = []

    to_drop = []
    noise_count = 0

    for i in range(len(df)-1):
        magnitude = sqrt(dx_dt[i]**2 + dy_dt[i]**2)

        # remove noisy data
        if magnitude >= human_thresh:
            to_drop.append(i)
            noise_count += 1
        elif magnitude > 0:
        # mark as saccade
            saccades.append((i,1))
        elif dx_dt < dt[i] and dy_dt < dt[i]:
            # mark as fixation
            fixations.append((i,1))

        plt.broken_barh(saccades, yrange=(0,1), facecolors='orange', label='saccade')
        plt.broken_barh(fixations, yrange=(0,1), facecolors='blue', label='fixations')
        plt.title('Saccades and Fixations During Trial')
        plt.legend()


    plt.show()

    new_df = new_df.drop(new_df.index[to_drop])

    print('Number of omissions due to noise:', noise_count)
    return new_df

def show_path(Ax, Ay):
    plt.scatter(Ax, Ay)
    plt.title('Angular Visual Movement (Degrees)')
    plt.show()

def show_angvel(Ax, Ay):

    #for t in range(len(Ax)):
    return None

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
    file_isabella = 'varjo_gaze_output_2021-02-10-17-22_isabella.csv' # sep=;
    file_james = 'varjo_gaze_output_2021-02-05_james.csv'  # sep=;
    file_mateusz = 'varjo_gaze_output_2021-02-05_mateusz.csv'  # sep=,
    file_marek = pd.read_spss('1_interpolated_degrees.sav')

    # read in file and format
    df = make_df(file_marek)

    # assign relevant data

    lx = df['LeftEyeForward_x']
    ly = df['LeftEyeForward_y']
    lz = df['LeftEyeForward_z']
    rx = df['RightEyeForward_x']
    ry = df['RightEyeForward_y']
    rz = df['RightEyeForward_z']
    #t = df['CaptureTime']/np.tile(100000),len(df)) # nanoseconds
    t = df['CaptureTime']

    # compute angular values
    df['Ax_left'] = np.rad2deg(np.arctan2(lx, lz))
    df['Ay_left'] = np.rad2deg(np.arctan2(ly, lz))
    df['Ax_right'] = np.rad2deg(np.arctan2(rx, rz))
    df['Ay_right'] = np.rad2deg(np.arctan2(ry, rz))

    # average visual angle between both eyes along each plane
    df['Avg_angular_x'] = df[['Ax_left', 'Ax_right']].mean(axis = 1)
    df['Avg_angular_y'] = df[['Ay_left', 'Ay_right']].mean(axis = 1)

    # show vision path in averaged angular degrees
    #show_path(df['Avg_angular_x'], df['Avg_angular_y'])
    print('Length of capture time:', len(df['CaptureTime']))
    print('Length of capture time differences:',
          len(np.diff(df['CaptureTime']/1000000)))

    # # show vision path, separately for each eye
    # plot_eye_path(df)

    # show angular displacement over time, averaged over both eyes
    #plot_vs_time(t, df['Avg_angular_x'], df['Avg_angular_y'])

    # plot angular velocity
    dt = np.diff(df['CaptureTime'])
    dx_dt = np.diff(df['Avg_angular_x'])/dt
    dy_dt = np.diff(df['Avg_angular_y'])/dt
    #plot_vs_time(df['CaptureTime'][:len(df)-1],dx_dt,dy_dt)

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


if __name__ == "__main__":
    # Testing
    # hello("Isabella")
    main()