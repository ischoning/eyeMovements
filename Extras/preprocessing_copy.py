# import sys
# sys.path.insert(1, "/Users/ischoning/PycharmProjects/eyeMovements/eventdetect-master")
import pandas as pd
from pylab import *
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import numpy as np
import scipy.stats as stats
import math
import matplotlib.transforms as mtransforms
import viterbi
import baum_welch
from scipy.interpolate import make_interp_spline, BSpline
from Sample import Sample
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import IsolationForest
import viterbi
import baum_welch

#matplotlib.use('macosx')
#plt.close('all')

def gaussian(mu, sigma, x):
    return 1.0/sqrt(2*pi*sigma**2)*exp(-(x-mu)**2/(2*sigma**2))

def clean_sequence(df):
    # determine the sequence of events
    prev_st = df.hidden_state[0]
    sequence = []
    count = 1
    for i in range(1, len(df)):
        st = df.hidden_state[i]
        if st == prev_st:
            count += 1
        elif st != prev_st:
            sequence.append([prev_st, count])
            count = 1
            prev_st = st
        if i == len(df) - 1:
            sequence.append([prev_st, count])

    print("Sequence:", sequence)

    # remove sequence if it contains fewer than 4 data points
    # Note: not possible to have two contiguous saccades without a fixation (or smooth pursuit)
    # in between
    s = np.array(sequence)
    s = s[:,1].astype(int)
    indices = np.where(s < 4)
    for i in indices[0]:
        start = np.sum(s[:i])
        end = np.sum(s[:i+1])
        df.drop(df.index[start:end], axis = 0, inplace = True)
        del sequence[i]
    df.reset_index(drop=True, inplace=True)

    print("Sequence after removing events less than 4 samples long:", sequence)

    return df


def label_event(event, states):
    if len(states) == 0:
        print("No states provided.")
        return
    if event == 1:
        return states[1] # 'Fixation'
    if event == 0:
        return states[0] # 'Saccade'
    return 'Other'

def print_events(t, event, states):
    events = []
    event_label = label_event(event.iloc[0], states)
    start = t.iloc[0]
    n = 1
    event_change = False
    for i in range(len(event)):
        if i == len(event)-1 or event.iloc[i] != event.iloc[i+1]:
            event_change = True
        if event_change:
            end = t.iloc[i]
            events.append([event_label, n, start, end])
            if i != len(event)-1:
                start = t.iloc[i+1]
                event_label = label_event(event.iloc[i+1], states)
                n = 1
                event_change = False
        else:
            n += 1

    print('=====================================================================')

    for e in events:
        print("%s for %d samples from %d ms to %d ms." % (e[0], e[1], e[2], e[3]))

    print('=====================================================================')

    print('Number of Fixation events:', sum(event==1))
    print('Number of Saccade events:', sum(event==0))
    print('Total number of events:', len(event))

    return None


def plot(df):
    # left eye
    X = np.array(df.d_l).reshape(-1,1)
    y = np.array(df.vel_l).reshape(-1,1)

    plt.scatter(df.d_l, df.vel_l, s = 0.5)
    # A = np.linspace(np.min(df.d_l), np.max(df.d_l), 100)
    # D = 21+2.2*A
    # plt.plot(A, D, ':', color = 'orange', label = 'Carpenters Thm: D = 21 + 2.2*A')
    # identify outliers in the training dataset
    iso = IsolationForest(contamination=0.1)
    yhat = iso.fit_predict(X)
    # select all rows that are not outliers
    mask = yhat != -1
    X_train, y_train = X[mask, :], y[mask]
    # fit the model
    model = LinearRegression()
    model.fit(X_train, y_train)
    # evaluate the model
    y_left = model.predict(X)
    plt.plot(X, y_left, color="green", label = 'best fit')
    plt.title('Left Eye: Amplitude vs Velocity')
    plt.xlabel('deg')
    plt.ylabel('deg/s')
    plt.legend()
    plt.show()
    plt.scatter(df.vel_l, df.accel_l, s=0.5)
    plt.title('Left Eye: Velocity vs Acceleration')
    plt.xlabel('deg/s')
    plt.ylabel('deg/s^2')
    plt.ylim((0,7500))
    plt.xlim((0,400))
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
    X = np.array(df.d_r).reshape(-1,1)
    y = np.array(df.vel_r).reshape(-1,1)

    plt.scatter(df.d_r, df.vel_r, s=0.5)
    # A = np.linspace(np.min(df.d_r), np.max(df.d_r), 100)
    # D = 21+2.2*A
    # plt.plot(A, D, ':', color = 'orange', label = 'Carpenters Thm: D = 21 + 2.2*A')
    # identify outliers in the training dataset
    iso = IsolationForest(contamination=0.1)
    yhat = iso.fit_predict(X)
    # select all rows that are not outliers
    mask = yhat != -1
    X_train, y_train = X[mask, :], y[mask]
    # fit the model
    model = LinearRegression()
    model.fit(X_train, y_train)
    # evaluate the model
    y_right = model.predict(X)
    plt.plot(df.d_r, y_right, color="green", label='best fit')
    plt.title('Right Eye: Amplitude vs Velocity')
    plt.xlabel('deg')
    plt.ylabel('deg/s')
    plt.legend()
    plt.show()
    plt.scatter(df.vel_r, df.accel_r, s=0.5)
    plt.title('Right Eye: Velocity vs Acceleration')
    plt.xlabel('deg/s')
    plt.ylabel('deg/s^2')
    plt.ylim((0,7500))
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

    return {'Left':y_left, 'Right':y_right}

# files
file = '/Users/ischoning/PycharmProjects/GitHub/data/varjo_events_4_0_0.txt'
file_np = '/Users/ischoning/PycharmProjects/GitHub/data/dpi_events_1.npy'
file_np = np.load(file_np)

# create dataframe
df = pd.read_csv(file, sep="\t", float_precision=None)
#df = pd.DataFrame(file_np)

df = df[['time', 'dt', 'device_time', 'left_pupil_measure1', 'right_pupil_measure1', 'target_angle_x', 'target_angle_y', 'right_gaze_x',
         'right_gaze_y', 'left_angle_x', 'left_angle_y', 'right_angle_x', 'right_angle_y']]

print(len(df))
# print(df.columns.values)
# print(df.dtypes)
# print(df.time.head())
# print(df.dt.head())
# print(df.device_time.head())  # same as time

# sample rate: 100 samples per sec
df = df[:1000]

plt.scatter(df.right_angle_x, df.right_angle_y, label='sample right', s = 0.5)
plt.scatter(df.left_angle_x, df.left_angle_y, label='sample left', s = 0.5, color = 'green')
plt.scatter(df.target_angle_x, df.target_angle_y, color='red', label='target', s = 0.5)
plt.title('Pre-Cleaning')
plt.xlabel('x (deg)')
plt.ylabel('y (deg)')
plt.legend()
plt.show()

# Remove NA's

df.replace([np.inf, -np.inf], np.nan)
df.dropna(inplace=True)

print(len(df))

# Calculate amplitude, velocity, acceleration, change in acceleration

del_x_r = np.diff(df.right_angle_x, prepend=0)
del_y_r = np.diff(df.right_angle_y, prepend=0)
df['d_r'] = np.sqrt(del_x_r ** 2 + del_y_r ** 2)

del_x_l = np.diff(df.left_angle_x, prepend=0)
del_y_l = np.diff(df.left_angle_y, prepend=0)
df['d_l'] = np.sqrt(del_x_l ** 2 + del_y_l ** 2)

df['isi'] = np.diff(df.time, prepend=0)
print(df.isi.head())

df['vel_r'] = df.d_r / df.isi
df['vel_l'] = df.d_l / df.isi

del_vel_r = np.diff(df.vel_r, prepend=0)
del_vel_l = np.diff(df.vel_l, prepend=0)

df['accel_r'] = abs(del_vel_r) / df.isi
df['accel_l'] = abs(del_vel_l) / df.isi

del_accel_r = np.diff(df.accel_r, prepend=0)
del_accel_l = np.diff(df.accel_l, prepend=0)

df['jolt_r'] = del_accel_r / df.isi
df['jolt_l'] = del_accel_l / df.isi

# remove the first three datapoints (due to intersample calculations)
df = df[3:]
df.reset_index(drop=True, inplace=True)

best_fit = plot(df)

# compute pairwise correlation matrix for review ## TOO MUCH COMPUTER POWER NEEDED

# # # prep for correlation between pupil sizes (left and right)
# corr_df = df[['left_pupil_measure1', 'right_pupil_measure1']]
# pe_corr = corr_df.corr('pearson')
# sp_corr = corr_df.corr('spearman')
# print(pe_corr)
# print(sp_corr)
#
# # prep for correlation between amplitude and veloctiy
# corr_df = df[['d_r', 'vel_r']]
# pe_corr = corr_df.corr('pearson')
# sp_corr = corr_df.corr('spearman')
# print(pe_corr)
# print(sp_corr)

# Clean data by eye physiology

print('num left pupil == 0:', np.sum(np.where(df.left_pupil_measure1 == 0)))
print('num right pupil == 0:', np.sum(np.where(df.right_pupil_measure1 == 0)))
print('num left vel == 0:', np.sum(np.where(df.vel_l == 0)))
print('num right vel == 0:', np.sum(np.where(df.vel_r == 0)))
print('-------------')
print('Correlation:')
# scores are between -1 and 1 for perfectly negatively correlated variables and
# perfectly positively correlated respectively (+- 0.5 is good threshold)

# pearson correlation
# Assumes Gaussian distribution of the data.
print('pearson:')
print('left eye amp vs vel:', stats.pearsonr(df.d_l, df.vel_l))
print('right eye amp vs vel:', stats.pearsonr(df.d_r, df.vel_r))
print('left eye vel vs accel:', stats.pearsonr(df.vel_l, df.accel_l))
print('right eye vel vs accel:', stats.pearsonr(df.vel_r, df.accel_r))

# spearman correlation (non-parametric rank-based approach)
# does not assume linear relation or gaussian distribution (also monotonic relation assumed)
print('spearman:')
print('left eye amp vs vel:', stats.spearmanr(df.d_l, df.vel_l))
print('right eye amp vs vel:', stats.spearmanr(df.d_r, df.vel_r))
print('left eye vel vs accel:', stats.spearmanr(df.vel_l, df.accel_l))
print('right eye vel vs accel:', stats.spearmanr(df.vel_r, df.accel_r))

print('-------------')

df['corr_vel_accel_l'] = cov(df.vel_l, df.accel_l)[0][1]/(std(df.vel_l)*std(df.accel_l))
df['corr_vel_accel_r'] = cov(df.vel_r, df.accel_r)[0][1]/(std(df.vel_r)*std(df.accel_r))

# initialize class
sample = Sample(df)

bad_data = []
cond_1, cond_2, cond_3, cond_4, cond_5 = 0, 0, 0, 0, 0
for i in range(1, len(df)-2):

    prev, current, next = sample.get_window(i)
    delta = 0.1

    # no pupil size?
    if current['Left']['pupil'] == 0 or current['Right']['pupil'] == 0:
        bad_data.append(i)
        cond_1 += 1

    # angular velocity greater than 1000 deg/s?
    elif current['Left']['vel'] > 1000 or current['Right']['vel'] > 1000:
        bad_data.append(i)
        cond_3 += 1

    # correlation between amplitude of movement and velocity
    # Are there sudden changes in velocity without change in position or vice versa?
    elif abs(best_fit['Left'][i] - current['Left']['vel'])/best_fit['Left'][i] > delta or abs(best_fit['Right'][i] - current['Right']['vel'])/best_fit['Right'][i] > delta:
        bad_data.append(i)
        cond_4 += 1

    # correlation between velocity and acceleration
    # Are there sudden changes in acceleration without change in position or velocity?
    # elif current['Left']['vel']:
    #     bad_data.append(i)
    #     cond_5 += 1

print("Number of datapoints with no pupil size:", cond_1)
print("Intersample velocity equal to zero or greater than 1000 deg/ms:", cond_3)
print("Outliers:", cond_4)

print('len of original data:', len(df))
print("len of 'bad data':", len(bad_data))
#df = df.iloc[1:]

df.drop(index = bad_data, inplace = True)
df.reset_index(drop=True, inplace=True)

plt.scatter(df.right_angle_x, df.right_angle_y, label='sample right', s = 0.5)
plt.scatter(df.left_angle_x, df.left_angle_y, label='sample left', s = 0.5, color = 'green')
plt.scatter(df.target_angle_x, df.target_angle_y, color='red', label='target', s = 0.5)
plt.title('After Cleaning')
plt.xlabel('x (deg)')
plt.ylabel('y (deg)')
plt.legend()
plt.show()

plt.plot(df.time, df.right_angle_x, label='x')
plt.plot(df.time, df.right_angle_y, label='y', color='red')
plt.plot(df.time, df.d_r, label='intersample distance traveled', color='green')
plt.legend()
plt.xlabel('time')
plt.ylabel('deg')
plt.title('Right Eye: Position vs Time')
plt.show()

# if velocity is greater than 3 standard deviations from the mean of the pmf, classify the point as saccade, else fixation
# NOTE that the white space in the plot is due to jump in ms between events
states = ['Saccade', 'Fixation']
df['fix1 sac0'] = np.where(df.vel_l <= 0.02, 1, 0)
event = df['fix1 sac0']

print('=============== STEP 1: Filter Saccades ===============')

# estimate priors (sample means)
mean_fix = np.mean(df[event == 1]['vel_l'])
mean_sac = np.mean(df[event == 0]['vel_l'])
std_fix = np.std(df[event == 1]['vel_l'])
std_sac = np.std(df[event == 0]['vel_l'])
print("Fixation: mean =", mean_fix, "standard deviation =", std_fix)
print("Saccade: mean =", mean_sac, "standard deviation =", std_sac)

print('\n============== BEGIN VITERBI ==============')
# first run EM to get best match params (priors, trans, emission probabilities)
# then run Viterbi HMM algorithm to output the most likely sequence given the params calculated in EM
obs = df.vel_l.astype(str)
obs = obs.tolist()
states = ['Sac', 'Fix']
start_p = [0.5, 0.5]
trans_p = np.array([[0.5, 0.5],
                    [0.5, 0.5]])
Sac = []
Fix = []
for o in obs:
    x = float(o)
    if o not in Sac:
        Sac.append(gaussian(mean_sac, std_sac, x))
    if o not in Fix:
        Fix.append(gaussian(mean_fix, std_fix, x))

emit_p = np.array([Sac, Fix])
df['hidden_state'] = viterbi.run(obs, states, start_p, trans_p, emit_p)
print(df['hidden_state'].value_counts())
print('=============== END VITERBI ===============')


print('\n============== BEGIN BAUM-WELCH ==============')
trans_p, emit_p = baum_welch.run(obs, states, start_p, trans_p, emit_p)
print('============== END BAUM-WELCH ==============')


print('\n============== BEGIN UPDATED VITERBI ==============')
df['hidden_state'] = viterbi.run(obs, states, start_p, trans_p, emit_p)
print('=============== END UPDATED VITERBI ===============')

df = clean_sequence(df)

print('\n=============== STEP 2: Classify Fixations and Smooth Pursuits ===============')

# filter out Saccades
df = df[df.hidden_state != 'Sac']
df.reset_index(drop=True, inplace=True)

ang_vel = df['ang_vel']
states = ['Smooth Pursuit', 'Fixation']
plt.plot(df.time, df.vel_l)
plt.show()

df['fix1 smp0'] = np.where(df.vel_l <= 0.02, 1, 0)
event = df['fix1 smp0']
print_events(df.time, event=event, states=states)

# estimate priors (sample means)
mean_fix = np.mean(df[event == 1]['vel_l'])
mean_smp = np.mean(df[event == 0]['vel_l'])
std_fix = np.std(df[event == 1]['vel_l'])
std_smp = np.std(df[event == 0]['vel_l'])
print("Fixation: mean =", mean_fix, "standard deviation =", std_fix)
print("Smooth Pursuit: mean =", mean_smp, "standard deviation =", std_smp)

print('\n============== BEGIN VITERBI ==============')

# first run EM to get best match params (priors, trans, emission probabilities)
# then run Viterbi HMM algorithm to output the most likely sequence given the params calculated in EM
obs = ang_vel.astype(str)
obs = obs.tolist()
states = ['SmP', 'Fix']
# p = math.log(0.5)
start_p = [0.5, 0.5]
trans_p = np.array([[0.5, 0.5],
                    [0.5, 0.5]])
# Note: not possible to have two contiguous saccades without a fixation (or smooth pursuit)
# in between
SmP = []
Fix = []
for o in obs:
    x = float(o)
    if o not in SmP:
        SmP.append(gaussian(mean_sac, std_sac, x))
    if o not in Fix:
        Fix.append(gaussian(mean_fix, std_fix, x))

emit_p = np.array([SmP, Fix])
df['hidden_state'] = viterbi.run(obs, states, start_p, trans_p, emit_p)
print(df['hidden_state'].value_counts())

print('=============== END VITERBI ===============')

print('\n============== BEGIN BAUM-WELCH ==============')

trans_p, emit_p = baum_welch.run(obs, states, start_p, trans_p, emit_p)

print('============== END BAUM-WELCH ==============')

print('\n============== BEGIN UPDATED VITERBI ==============')

df['hidden_state'] = viterbi.run(obs, states, start_p, trans_p, emit_p)
# print(len(df['hidden_state']))
print(df['hidden_state'].value_counts())

print('=============== END UPDATED VITERBI ===============')
#plot(df)

# bad_data = []
# cond_1, cond_2, cond_3, cond_4, cond_5 = 0, 0, 0, 0, 0
# for i in range(len(df)):
    # d_l = df.d_l[i]
    # d_r = df.d_r[i]
    # vel_l = df.vel_l[i]
    # vel_r = df.vel_r[i]
    # accel_l = df.accel_l[i]
    # accel_r = df.accel_r[i]
    # pupil_l = df.left_pupil_measure1[i]
    # pupil_r = df.right_pupil_measure1[i]
    # """
    # print('cond_1:')
    # print('    pupil_l:', pupil_l)
    # print('    pupil_r:', pupil_r)
    # print('cond_2:')
    # print('    abs(pupil_l - pupil_r):', abs(pupil_l - pupil_r))
    # print('cond_3:')
    # print('    vel_l:', vel_l)
    # print('    vel_r:', vel_r)
    # print('cond_4:')
    # print('    abs(vel_l - 21 - 2.2*d_l):', abs(vel_l - 21 - 2.2*d_l))
    # print('    abs(vel_r - 21 - 2.2*d_r):', abs(vel_r - 21 - 2.2*d_r))
    # print('cond_5:')
    # print('    abs(accel_l - 21 - 2.2*vel_l):', abs(accel_l - 21 - 2.2*vel_l))
    # print('    abs(accel_r - 21 - 2.2*vel_r):', abs(accel_r - 21 - 2.2*vel_r))
    # print('correlation:')
    # print('    spearmanr(d_l, vel_l):', stats.spearmanr(d_l, vel_l))
    # print('    spearmanr(d_r, vel_r):', stats.spearmanr(d_r, vel_r))
    # print('    spearmanr(vel_l, accel_l):', stats.spearmanr(vel_l, accel_l))
    # print('    spearmanr(vel_r, accel_r):', stats.spearmanr(vel_r, accel_r))
    # """
    # no pupil size?
    # if pupil_l == 0 or pupil_r == 0:
    #     bad_data.append(i)
    #     cond_1 += 1

    # pupil size correlated?
    # elif abs(pupil_l - pupil_r) > 0.005:
    #     bad_data.append(i)
    #     cond_2 += 1

    # angular velocity greater than 1000 deg/s?
    # elif vel_r > 1000 or vel_l > 1000:
    #     bad_data.append(i)
    #     cond_3 += 1

    # correlation between amplitude of movement and velocity agrees with
    # Carpenter's Theorem: D = 21 + 2.2A where D~deg/ms, A~deg traversed
    # Find: margin of error (plot histogram of error between predicted and actual D or A.
    # Are there 2 distributions or outliers?)
    # elif abs(vel_l - 2.2*d_l) > 21 or abs(vel_r - 2.2*d_r) > 21:
    #     bad_data.append(i)
    #     cond_4 += 1

    # changes of direction in velocity during saccade (velocity and acceleration correlated)
    # >> I don't know how to check this element-wise >> Maybe try: Does area under accel curve agree with vel?
    # >>> if error (a_meas - a_pred)/a_pred > threshold (5%) then bad data.
    # elif np.where(abs(df.corr_vel_accel_l) < 0.5) == 1 or np.where(abs(df.corr_vel_accel_r) < 0.5) == 1:
    #     bad_data.append(i)
    #     cond_5 += 1

# print("Number of datapoints with no pupil size:", cond_1)
# print("Number of datapoints with uncorrelated pupil sizes:", cond_2)
# print("Number of datapoints with intersample velocity equal to zero or greater than 1000 deg/ms:", cond_3)
# print("Number of datapoints with no correlation (cov<0.5) between displacement and velocity:", cond_4)
# print("Number of datapoints with no correlation (cov<0.5) between velocity and acceleration:", cond_5)

# print('len of original data:', len(df))
# print("len of 'bad data':", len(bad_data))
# #df = df.iloc[1:]
#
# df.drop(index = bad_data, inplace = True)
# df.reset_index(drop=True, inplace=True)
#
# plot(df)