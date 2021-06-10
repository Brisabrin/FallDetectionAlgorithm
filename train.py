import pandas as pd
import numpy as np

from scipy.spatial import distance
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

df = pd.read_csv('CompleteDataSet.csv')

n, m = df.shape
x = []
y = []

ind = df.columns.get_loc('Subject')
sInd = df.columns.get_loc('BeltAccelerometer')


subject = 0
activity = 0
trial = 0


print(df.iloc[:, ind + 1])
for i in range(1, n):

    # subject
    a = df.iloc[i, ind]

    # activity

    b = df.iloc[i, ind + 1]

    # trial

    c = df.iloc[i, ind + 2]

    if subject == a and activity == b and trial == c:
        x[len(x) - 1].append(df.iloc[i, sInd])

    else:
        x.append([df.iloc[i, sInd]])
        act = df.columns.get_loc('Activity')
        y.append(df.iloc[i, act])
        subject = a
        activity = b
        trial = c

# print(x[0])


X_train, X_test, y_train, y_test = train_test_split(
    x, y, test_size=0.33, random_state=42)


def DTW(a, b):
    an = a.size
    bn = b.size
    pointwise_distance = distance.cdist(a.reshape(-1, 1), b.reshape(-1, 1))
    cumdist = np.matrix(np.ones((an+1, bn+1)) * np.inf)
    cumdist[0, 0] = 0

    for ai in range(an):
        for bi in range(bn):
            minimum_cost = np.min([cumdist[ai, bi+1],
                                   cumdist[ai+1, bi],
                                   cumdist[ai, bi]])
            cumdist[ai+1, bi+1] = pointwise_distance[ai, bi] + minimum_cost

    return cumdist[an, bn]


# parameters = {'n_neighbors': [2, 4, 8]}
clf = KNeighborsClassifier(metric=DTW, n_neighbors=11)
clf.fit(X_train, y_train)

# evaluate
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))
