import os
import glob
import pickle
import numpy as np
import pandas as pd
from scipy import signal, interpolate
# import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


def filter_data(X, Y, Z, fs, fc):
    w = fc / (fs / 2)
    b, a = signal.butter(fc, w, 'low', output='ba')

    X_filt = signal.filtfilt(b, a, X)
    Y_filt = signal.filtfilt(b, a, Y)
    Z_filt = signal.filtfilt(b, a, Z)

    return X_filt, Y_filt, Z_filt


# cut factor of the original data at the beginning
def preCut(X, Y, Z, Time, precutfactor):
    indx = np.round((precutfactor * len(Time))).astype(int)
    XX = X[indx:]
    YY = Y[indx:]
    ZZ = Z[indx:]
    TTime = Time[indx:]

    return XX, YY, ZZ, TTime


def preCalcPCA(X, freq):
    mean = np.mean(X)

    peaks, _ = signal.find_peaks(X, height=(mean, mean + 50), distance=freq / 3)
    heightUpp = np.max(X[peaks])
    heightLow = np.median(X)

    return heightUpp, heightLow


def preCalc(X, Y, Z, freq, time, leftB, rightB):
    start = np.round((leftB * len(time))).astype(int)
    end = np.round((rightB * len(time))).astype(int)
    XX = X[start:end]
    YY = Y[start:end]
    ZZ = Z[start:end]

    xyzset = np.column_stack([XX, YY, ZZ])
    xyz_abs = np.linalg.norm(xyzset, axis=1)
    mean = np.mean(xyz_abs)

    peaks, _ = signal.find_peaks(xyz_abs, height=(mean, mean + 50), distance=freq / 3)
    heightUpp = np.max(xyz_abs[peaks])
    heightLow = np.median(xyz_abs)

    # time0 = time[start:end]
    # plt.figure(figsize=(6, 4), dpi=90)
    # plt.plot(time0, xyz_abs, label='xyz_abs')
    # plt.scatter(time0[peaks], xyz_abs[peaks], c='r', label='xyz_abs peaks')
    # plt.legend(loc='best')

    return heightUpp, heightLow


def cut_data(Xa, Ya, Za, Xg, Yg, Zg, freq1, freq2, heightUppA, heightLowA,
             heightUppG, heightLowG, timeA, timeG, modeTime, cutPeakFactor):
    acc_filt = np.column_stack([Xa, Ya, Za])
    acc_abs = np.linalg.norm(acc_filt, axis=1)
    gyr_filt = np.column_stack([Xg, Yg, Zg])
    gyr_abs = np.linalg.norm(gyr_filt, axis=1)

    peaks1, _ = signal.find_peaks(acc_abs, prominence=0.2 * (heightUppA - heightLowA),
                                  height=(heightLowA, heightUppA), distance=freq1 / 3)
    indx1 = np.round((cutPeakFactor * len(peaks1))).astype(int)
    diff_peaks1 = np.diff(peaks1)
    gap11 = np.argmax(diff_peaks1[:indx1])

    # take mode seconds from start
    acc_end = peaks1[gap11 + 1] + freq1 * modeTime
    acc_cut = acc_filt[peaks1[gap11 + 1]:acc_end.astype(int), :]
    time_cutA = timeA[peaks1[gap11 + 1]:acc_end.astype(int)]

    # plt.figure(figsize=(6, 4), dpi=90)
    # plt.plot(TimeA, acc_abs, label='acc_abs')
    # plt.scatter(TimeA[peaks1], acc_abs[peaks1], c='r', label='acc_abs peaks')
    # plt.plot(time_cutA, acc_cut[:, 2], c='g', label='acc_cut')
    # plt.legend(loc='best')

    peaks2, _ = signal.find_peaks(gyr_abs, prominence=0.2 * (heightUppG - heightLowG),
                                  height=(heightLowG, heightUppG), distance=freq2 / 3)
    indx2 = np.round((cutPeakFactor * len(peaks2))).astype(int)
    diff_peaks2 = np.diff(peaks2)
    gap12 = np.argmax(diff_peaks2[:indx2])

    # take mode seconds from start
    gyr_end = peaks2[gap12 + 1] + freq2 * modeTime
    gyr_cut = gyr_filt[peaks2[gap12 + 1]:gyr_end.astype(int), :]
    time_cutG = timeG[peaks2[gap12 + 1]:gyr_end.astype(int)]

    # plt.figure(figsize=(6, 4), dpi=90)
    # plt.plot(TimeG, gyr_abs, label='gyr_abs')
    # plt.scatter(TimeG[peaks2], gyr_abs[peaks2], c='r', label='gyr_abs peaks')
    # plt.plot(time_cutG, gyr_cut[:, 1], c='g', label='gyr_cut')
    # plt.legend(loc='best')

    return acc_cut, gyr_cut, time_cutA, time_cutG


def pca_data(data_cut):
    pca = PCA(n_components=3)
    data_pca = pca.fit_transform(data_cut)
    x_pca = data_pca[:, 0]
    y_pca = data_pca[:, 1]
    z_pca = data_pca[:, 2]

    return x_pca, y_pca, z_pca


# find all peaks after pca and resample between neighbor peaks
def resample_dataA(a_pca0, time_cutA0, sampling_frequency1, heightUppA, heightLowA):
    time_cutA, indx = np.unique(time_cutA0, return_index=True)
    a_pca = a_pca0[indx]
    peaks_a, _ = signal.find_peaks(a_pca, prominence=0.5 * (heightUppA - heightLowA), height=(heightLowA, heightUppA),
                                   distance=sampling_frequency1 / 2)

    num = 100
    sampleA = np.zeros((len(peaks_a) - 1, num))
    timelineA = np.zeros_like(sampleA)
    for i in range(len(peaks_a) - 1):
        start = peaks_a[i]
        end = peaks_a[i + 1]
        fA = interpolate.interp1d(time_cutA[start:(end + 1)], a_pca[start:(end + 1)], kind='cubic', axis=- 1)
        timelineA[i] = np.linspace(time_cutA[start], time_cutA[end], num)
        sampleA[i] = fA(timelineA[i])

    # plt.figure(figsize=(6, 4), dpi=90)
    # plt.plot(time, a_pca, label='PCA xyz_abs')
    # plt.scatter(time[peaks_a], a_pca[peaks_a], c='r', label='PCA xyz_abs peaks')
    # plt.legend(loc='best')

    return sampleA


# find all peaks after pca and resample between neighbor peaks
def resample_dataG(g_pca0, time_cutG0, sampling_frequency2, heightUppG, heightLowG):
    time_cutG, indx = np.unique(time_cutG0, return_index=True)
    g_pca = g_pca0[indx]
    peaks_g, _ = signal.find_peaks(g_pca, prominence=0.5 * (heightUppG - heightLowG), height=(heightLowG, heightUppG),
                                   distance=sampling_frequency2 / 2)

    num = 100
    sampleG = np.zeros((len(peaks_g) - 1, num))
    timelineG = np.zeros_like(sampleG)
    for i in range(len(peaks_g) - 1):
        start = peaks_g[i]
        end = peaks_g[i + 1]
        fG = interpolate.interp1d(time_cutG[start:(end + 1)], g_pca[start:(end + 1)], kind='cubic', axis=- 1)
        timelineG[i] = np.linspace(time_cutG[start], time_cutG[end], num)
        sampleG[i] = fG(timelineG[i])

    return sampleG


def removeSequence(matrix0):
    if np.shape(matrix0)[0] < 4:
        return matrix0
    else:
        average = np.mean(abs(matrix0), axis=1)
        sd = np.std(average)
        median1 = np.mean(average)
        c = 0
        tmp = np.array([], dtype=int)
        for i in average:
            if (i > (median1 + (2 * sd))) or (i < (median1 - (2 * sd))):
                tmp = np.append(tmp, c)
                c += 1

        matrix = np.delete(matrix0, tmp, axis=0)

        return matrix


def dataprocess(Xa, Ya, Za, Xg, Yg, Zg, timeA, timeG, modeTime_in, preCutFactor, cutPeakFactor, leftB, rightB):
    sampling_frequency1 = np.round(len(timeA) / (timeA[-1]))
    sampling_frequency2 = np.round(len(timeG) / (timeG[-1]))

    # pre-calc: calculate the bounds of height for signal.find_peaks()
    heightUppA, heightLowA = preCalc(Xa, Ya, Za, sampling_frequency1, timeA, leftB, rightB)
    heightUppG, heightLowG = preCalc(Xg, Yg, Zg, sampling_frequency2, timeG, leftB, rightB)

    # filter
    Xa_filtered, Ya_filtered, Za_filtered = filter_data(Xa, Ya, Za, fs=sampling_frequency1, fc=4)
    Xg_filtered, Yg_filtered, Zg_filtered = filter_data(Xg, Yg, Zg, fs=sampling_frequency2, fc=4)

    # pre-cut
    Xa_preCut, Ya_preCut, Za_preCut, timeA_preCut = preCut(Xa_filtered, Ya_filtered, Za_filtered, timeA,
                                                           preCutFactor)
    Xg_preCut, Yg_preCut, Zg_preCut, timeG_preCut = preCut(Xg_filtered, Yg_filtered, Zg_filtered, timeG,
                                                           preCutFactor)

    # cut
    acc_cut, gyr_cut, timeA_cut, timeG_cut = cut_data(Xa_preCut, Ya_preCut, Za_preCut,
                                                      Xg_preCut, Yg_preCut, Zg_preCut,
                                                      sampling_frequency1, sampling_frequency2,
                                                      heightUppA, heightLowA, heightUppG, heightLowG,
                                                      timeA_preCut, timeG_preCut, modeTime_in, cutPeakFactor)

    # PCA                                                 
    xa_pca, ya_pca, za_pca = pca_data(acc_cut)
    xg_pca, yg_pca, zg_pca = pca_data(gyr_cut)

    # pre-calc: calculate the bounds of height for signal.find_peaks()
    heightUppA_pca1, heightLowA_pca1 = preCalcPCA(xa_pca, sampling_frequency1)
    heightUppA_pca2, heightLowA_pca2 = preCalcPCA(ya_pca, sampling_frequency1)
    heightUppA_pca3, heightLowA_pca3 = preCalcPCA(za_pca, sampling_frequency1)
    heightUppG_pca1, heightLowG_pca1 = preCalcPCA(xg_pca, sampling_frequency2)
    heightUppG_pca2, heightLowG_pca2 = preCalcPCA(yg_pca, sampling_frequency2)
    heightUppG_pca3, heightLowG_pca3 = preCalcPCA(zg_pca, sampling_frequency2)

    # resample here, all sample and timeline are matrix
    xa_sample = resample_dataA(xa_pca, timeA_cut, sampling_frequency1, heightUppA_pca1, heightLowA_pca1)
    ya_sample = resample_dataA(ya_pca, timeA_cut, sampling_frequency1, heightUppA_pca2, heightLowA_pca2)
    za_sample = resample_dataA(za_pca, timeA_cut, sampling_frequency1, heightUppA_pca3, heightLowA_pca3)
    xg_sample = resample_dataG(xg_pca, timeG_cut, sampling_frequency2, heightUppG_pca1, heightLowG_pca1)
    yg_sample = resample_dataG(yg_pca, timeG_cut, sampling_frequency2, heightUppG_pca2, heightLowG_pca2)
    zg_sample = resample_dataG(zg_pca, timeG_cut, sampling_frequency2, heightUppG_pca3, heightLowG_pca3)

    return xa_sample, ya_sample, za_sample, xg_sample, yg_sample, zg_sample


def targetget(string):
    if 'downstairs' in string:
        targettensor = [0, 0, 1]
    elif 'normal' in string:
        targettensor = [0, 1, 0]
    else:
        targettensor = [1, 0, 0]

    return targettensor


def readhw(sub, hw):
    Height = hw['height [m]']
    Weight = hw['weight [kg]']

    sub_index = hw[hw['subject no.'] == sub].index
    height_read = Height[sub_index]
    weight_read = Weight[sub_index]
    heightlist = height_read.tolist()
    weightlist = weight_read.tolist()

    if len(heightlist) == 0:
        heightn = 0
        weightn = 0
    else:
        heightn = heightlist[0]
        weightn = weightlist[0]

    return heightn, weightn


if __name__ == '__main__':
    path = "C:/Users/JINYANG/Downloads/Chrome/RawData/Smartphone3/"
    HW = pd.read_csv('C:/Users/JINYANG/Downloads/Chrome/RawData/anthro_2021.csv')

    subjects = os.listdir(path)
    threshold = 100

    input_data = []
    target_data = []

    for subject in subjects:
        # first read each subject
        files = os.listdir(path + subject)
        subject_number = int(subject[7:10])

        height, weight = readhw(subject_number, HW)

        if 'normal' in str(subject):
            modetime = 15
            precutFactor = 0.30
            cutpeakFactor = 0.25
            leftBound = 0.55
            rightBound = 0.6
        elif 'downstairs' in str(subject):
            modetime = 5
            precutFactor = 0.20
            cutpeakFactor = 0.25
            leftBound = 0.4
            rightBound = 0.5
        elif 'upstairs' in str(subject):
            modetime = 5
            precutFactor = 0.20
            cutpeakFactor = 0.25
            leftBound = 0.4
            rightBound = 0.5
        elif 'impaired' in str(subject):
            continue
        elif height == 0 or weight == 0:
            continue

        print(subject)

        acc_file = glob.glob(path + subject + "/" + "Accelerometer.csv")
        gyr_file = glob.glob(path + subject + "/" + "Gyroscope.csv")

        data_acc = pd.read_csv(acc_file[0], sep=',', header=None)
        acc = data_acc.iloc[2:, 1:4].values.astype(float)
        TimeA = data_acc.iloc[2:, 0].values.astype(float)
        data_gyr = pd.read_csv(gyr_file[0], sep=',', header=None)
        gyr = data_gyr.iloc[2:, 1:4].values.astype(float)
        TimeG = data_gyr.iloc[2:, 0].values.astype(float)

        xa = acc[:, 0]
        ya = acc[:, 1]
        za = acc[:, 2]
        xg = gyr[:, 0]
        yg = gyr[:, 1]
        zg = gyr[:, 2]

        xa_processed, ya_processed, za_processed, xg_processed, yg_processed, zg_processed = \
            dataprocess(xa, ya, za, xg, yg, zg, TimeA, TimeG, modetime,
                        precutFactor, cutpeakFactor, leftBound, rightBound)

        # remove malicious sequences
        if np.shape(xa_processed)[0] == 0 or np.shape(ya_processed)[0] == 0 or np.shape(za_processed)[0] == 0 or \
                np.shape(xg_processed)[0] == 0 or np.shape(yg_processed)[0] == 0 or np.shape(zg_processed)[0] == 0:
            continue
        else:
            xa_clean = removeSequence(xa_processed)
            ya_clean = removeSequence(ya_processed)
            za_clean = removeSequence(za_processed)
            xg_clean = removeSequence(xg_processed)
            yg_clean = removeSequence(yg_processed)
            zg_clean = removeSequence(zg_processed)

        heights = np.ones((1, threshold)) * height
        weights = np.ones((1, threshold)) * weight
        input1 = np.concatenate([heights, weights], axis=0)

        input_acc = np.stack([xa_clean[0], ya_clean[0], za_clean[0]], axis=0)
        input_gyr = np.stack([xg_clean[0], yg_clean[0], zg_clean[0]], axis=0)
        input2 = np.concatenate([input_acc, input_gyr], axis=0)

        inputData = np.concatenate([input1, input2], axis=0)

        targetData = targetget(str(subject))

        input_data.append(inputData)
        target_data.append(targetData)

    input_data = np.array(input_data)
    target_data = np.array(target_data)
    print(input_data.shape)
    print(target_data.shape)

    f = open("input.pickle", 'wb+')
    pickle.dump(input_data, f)
    f1 = open("target.pickle", 'wb+')
    pickle.dump(target_data, f1)
