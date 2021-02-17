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
    df = data[['LeftEyeForward_x', 'LeftEyeForward_y', 'RightEyeForward_x', 'RightEyeForward_y']]

    # format to floats
    # replace scientific notation with 0
    for col in df:
        f = lambda x: x.apply(str).str.replace(".", "").str.contains('e', na=False, case=False)
        df1 = df.iloc[:, 0:].apply(f, axis=0)
        df.loc[(df1[col] == 1), col] = 0
        df[col] = df[col].astype(float)
        #print(df[col].isna().sum())

        #df.loc[df[col].str.contains("e", na=False, case=False), col] = 0
    #for col in df:
        #df[col] = pd.to_numeric(df[col], errors='coerce')
    #df.replace(np.nan, 0, regex=True)

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
    plt.title('Left Eye Movement in the X,Y Plane')
    plt.show()

    df.plot.scatter(x='RightEyeForward_x', y='RightEyeForward_y')
    plt.title('Right Eye Movement in the X,Y Plane')
    plt.show()

def plot_single_eye_vals(df):
    '''
    input: pandas dataframe that includes 'Left(Right)EyeForward_x(y)' in header
    output: two plots (x vs y movements of each eye)
    return: None
    '''
    lx = df['LeftEyeForward_x']
    ly = df['LeftEyeForward_y']

    # Left Eye x, y
    l = range(0, len(df['LeftEyeForward_x']))
    plt.plot(l, lx, 'r', label='x')
    plt.plot(l, ly, 'b', label='y')
    plt.title('Left Eye')
    plt.legend()
    #plt.fill([3, 4, 4, 3], [2, 2, 4, 4], 'b', alpha=0.2, edgecolor='r')
    plt.show()

    rx = df['RightEyeForward_x']
    ry = df['RightEyeForward_y']

    # Right Eye x, y
    r = range(0, len(df['RightEyeForward_x']))
    plt.plot(r, rx, 'r', label='x')
    plt.plot(r, ry, 'b', label='y')
    plt.title('Right Eye')
    plt.legend()
    plt.show()

    return None

if __name__ == "__main__":
    # Testing
    # hello("Isabella")

    # files
    file = 'varjo_gaze_output_2021-02-10-17-22_isabella.csv' # sep=;
    file_james = 'varjo_gaze_output_2021-02-05_james.csv'  # sep=;
    file_mateusz = 'varjo_gaze_output_2021-02-05_mateusz.csv'  # sep=,

    # read in file and format
    df = make_df(file)

    # make 2D scatter plots of single eye movements in x,y plane
    plot_eye_path(df)

    # plot 1D x and y movement of each eye
    plot_single_eye_vals(df)

    # next step: segment graph given a condition (display on plot and return locs)
    # next step: use LMS filter and HMM filter to identify fixation and saccade
    # compare results