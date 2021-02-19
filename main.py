import sys
sys.path.insert(1, "/Users/ischoning/PycharmProjects/eyeMovements/eventdetect-master")
import pandas as pd
from pylab import *
import matplotlib.pyplot as plt
import numpy as np

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
    df = data[['LeftEyeForward_x', 'LeftEyeForward_y', 'LeftEyeForward_z',
               'RightEyeForward_x', 'RightEyeForward_y', 'RightEyeForward_z']]

    # format to floats
    # replace scientific notation with 0
    for col in df:
        f = lambda x: x.apply(str).str.replace(".", "").str.contains('e', na=False, case=False)
        df1 = df.iloc[:, 0:].apply(f, axis=0)
        df.loc[(df1[col] == 1), col] = 0
        df[col] = df[col].astype(float)

    # check output
    print(df.head())
    print(df.dtypes)

    return df

def plot_eye_path(df):
    '''
    input: pandas dataframe that includes 'Left(Right)EyeForward_x(y)' in header
    output: two scatter plots (x,y movements of each eye)
    return: None
    '''
    df.plot.scatter(x='LeftEyeForward_x', y='LeftEyeForward_y')
    df.plot.scatter(x='RightEyeForward_x', y='RightEyeForward_y')
    plt.title('Left Eye Movement in the X,Y Plane')
    plt.show()

    df.plot.scatter(x='RightEyeForward_x', y='RightEyeForward_y')
    plt.title('Right Eye Movement in the X,Y Plane')
    plt.show()

def plots_vs_seq(x, y, x_label, y_label, title):
    '''
    input: two values (x,y) to be plotted against time
    output: two plots
    return: None
    '''

    # Left Eye x, y
    end = len(x) if len(x) > len(y) else len(y)
    l = range(0, end)

    plt.plot(l, x, 'r', label=x_label)
    plt.plot(l, y, 'b', label=y_label)
    plt.title(title)
    plt.legend()
    plt.show()

    return None

def delta_method(df):
    '''
    Uses predefined thresholds for event categorization, ie Identification by
     dispersion threshold (I-DT).
    A simple function that categorizes eye movement as fixation or
    saccade by analyzing change in location.
    input: pandas dataframe that includes 'Left(Right)EyeForward_x(y)' in header, angular flag
    output: two plots classifying fixations and saccades for each eye
    return: None
    '''

    fig, ax = plt.subplots(2,1,sharex=True)
    title = ('Left Eye Fixations and Saccades', 'Right Eye Fixations and Saccades')

    dt = 0.01

    for k in range(2):
        saccades = []
        fixations = []
        if k == 0:
            x = df['LeftEyeForward_x']
            y = df['LeftEyeForward_y']
        else:
            x = df['RightEyeForward_x']
            y = df['RightEyeForward_y']

        for i in range(1,len(df)-1):
            x0 = x[i-1]
            x1 = x[i]
            y0 = y[i-1]
            y1 = y[i]
            dx_dt = abs(x1-x0)/dt
            dy_dt = abs(y1-y0)/dt
            if dx_dt > 1 or dy_dt > 1:
                # mark as saccade
                saccades.append((i,1))
            if dx_dt < dt and dy_dt < dt:
                # mark as fixation
                fixations.append((i,1))

        ax[k].broken_barh(saccades, yrange=(0,1), facecolors='orange', label='saccade')
        ax[k].broken_barh(fixations, yrange=(0,1), facecolors='blue', label='fixations')
        ax[k].set_title(title[k])
        ax[k].legend()

    plt.show()

    return None


def main():

    # files
    file_isabella = 'varjo_gaze_output_2021-02-10-17-22_isabella.csv' # sep=;
    file_james = 'varjo_gaze_output_2021-02-05_james.csv'  # sep=;
    file_mateusz = 'varjo_gaze_output_2021-02-05_mateusz.csv'  # sep=,

    # read in file and format
    df = make_df(file_isabella)

    # assign relevant data
    lx = df['LeftEyeForward_x']
    ly = df['LeftEyeForward_y']
    lz = df['LeftEyeForward_z']
    rx = df['RightEyeForward_x']
    ry = df['RightEyeForward_y']
    rz = df['RightEyeForward_z']

    # compute angular values
    Ax_left = np.rad2deg(np.arctan2(lx, lz))
    Ay_left = np.rad2deg(np.arctan2(ly, lz))
    Ax_right = np.rad2deg(np.arctan2(rx, rz))
    Ay_right = np.rad2deg(np.arctan2(ry, rz))

    # average visual angle between both eyes along each plane
    Ax = average(Ax_left, Ax_right)
    Ay = average(Ay_left, Ay_right)
    df['Angular_x'] = Ax
    df['Angular_y'] = Ay

    # make 2D scatter plots of single eye movements in x,y plane
    plot_eye_path(df)

    # plot 1D x and y movement of each eye
    plots_vs_seq(lx, ly, 'x', 'y', 'Left Eye Coordinates Over Time')
    plots_vs_seq(rx, ry, 'x', 'y', 'Right Eye Coordinates Over Time')
    plots_vs_seq(lx, rx, 'left eye', 'right eye', 'Movement Along X-Axis')
    plots_vs_seq(ly, ry, 'left eye', 'right eye', 'Movement Along Y-Axis')

    # segment graph given a condition (display on plot and return locs)
    delta_method(df)


    # smooth x and y movement using data from both eye and probability theory,
    # ie minimize noise (EyeLink algorithm)

    # next step: use LMS filter and HMM filter to identify fixation and saccade
    # compare results. MarkEye?


if __name__ == "__main__":
    # Testing
    # hello("Isabella")
    main()