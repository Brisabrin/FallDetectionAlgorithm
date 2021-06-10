import numpy as np
import pandas as pd
from collections import defaultdict
import os
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sktime.classification.compose import ColumnEnsembleClassifier
from sktime.classification.dictionary_based import BOSSEnsemble
from sktime.classification.interval_based import TimeSeriesForestClassifier
# from sktime.classification.shapelet_based import MrSEQLClassifier
from sktime.datasets import load_basic_motions
from sktime.transformations.panel.compose import ColumnConcatenator
from sktime.transformations.panel.padder import PaddingTransformer
from pickle5 import pickle
import joblib
import struct


model = pickle.load(open('model.sav', 'rb'))

# Sensor type :acc = 0 , gyro = 1

co = 1

arr = []

feat = [" X-Axis", " Z-Axis"]


for i in range(0, 40):
    arr.append(i)

Y = []

acc = defaultdict(list)
gyro = defaultdict(list)

for subdir, dirs, files in os.walk("/Users/brisamaneechotesuwan/Desktop/Fall detection/UMAFall_Dataset"):
    for file in files:
        # print os.path.join(subdir, file)
        filepath = subdir + os.sep + file
        print(file)
        # df = pd.read_csv("UMAFall_Dataset/"+file, skiprows=arr, sep=';',
        #  header = None)

        s = pd.read_csv("UMAFall_Dataset/" + file,
                        header=None).iloc[8, :].values
        print(type(s))
        if "FALL" in s[0]:
            s = 'F'
        else:
            s = 'D'

        df = pd.read_csv(
            "UMAFall_Dataset/" + file, skiprows=arr, sep=';')

        print(s)

        # acc

        # i : sensor id , j  : sensor type
        for i in range(2, 3):
            track = False
            for j in range(2):
                f = df[(df[" Sensor Type"] == j) & (df[" Sensor ID"] == i)]

                n, m = f.shape
                if n == 0 or m == 0:
                    track = True
                    continue

                if j == 0:
                    res = 16
                    ra = 16 * 2

                else:
                    res = 16
                    ra = 256 * 2

                for l in range(2):

                    if j == 0:
                        acc[feat[l]].append(pd.Series(
                            [(z * round((2**res) / ((2 * ra)), 6)) for z in f.iloc[:, l + 2].values], dtype='object'))

                    else:
                        gyro[feat[l]].append(pd.Series
                                             ([(z * round((2**res) / ((2 * ra)), 6)) for z in f.iloc[:, l + 2].values], dtype='object'))

                if not track:
                    Y.append(s)

        co += 1
        if co > 1:
            break

# dic = {1: [1, 2, 3], 2: [3, 4, 5]}

# df = pd.DataFrame(dic)


# print(dic)
acc = pd.DataFrame(acc)
gyro = pd.DataFrame(gyro)
print(gyro)

frame = [acc, gyro]
X = pd.concat(frame, axis=1)

n, m = X.shape


pad = PaddingTransformer(pad_length=36000).fit(X)
X = pad.transform(X)

Y = pd.Series(Y)
print(X.shape)

# print(model.score(X, Y))

print(model.predict(X))
print("hello")
