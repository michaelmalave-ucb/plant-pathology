# from https://blog.hyperiondev.com/index.php/2019/02/18/machine-learning/
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import time

start_time = time.time()

# load our dataset
#train_data = scipy.io.loadmat('./data/train_32x32.mat')
train_data = scipy.io.loadmat('./data/extra_32x32.mat')

# extract the images and labels from the dictionary object
X = train_data['X']
y = train_data['y']

print("as loaded:")
print("X.shape = ", X.shape)
print("y.shape = ", y.shape)

# view an image (e.g. 25) and print its corresponding label
# img_index = 25
# plt.imshow(X[:,:,:,img_index])
# plt.show()
# print(y[img_index])

from sklearn.utils import shuffle
X = X.reshape(X.shape[0]*X.shape[1]*X.shape[2],X.shape[3]).T
y = y.reshape(y.shape[0],)
X, y = shuffle(X, y, random_state=42)

print("flattened:")
print("X.shape = ", X.shape)
print("y.shape = ", y.shape)

# print("reducing to small X and y for testing")
#
# small = 20000
# X = X[:small]
# y = y[:small]
#
# print("X.shape = ", X.shape)
# print("y.shape = ", y.shape)

from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier()
print(clf)

RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
           max_depth=None, max_features='auto', max_leaf_nodes=None,
           min_impurity_split=1e-07, min_samples_leaf=1,
           min_samples_split=2, min_weight_fraction_leaf=0.0,
           n_estimators=10, n_jobs=1, oob_score=False, random_state=None,
           verbose=2, warm_start=False)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
clf.fit(X_train, y_train)

print("after train_test_split:")
print("X_train.shape = ", X_train.shape)
print("y_train.shape = ", y_train.shape)
print("X_test.shape = ", X_test.shape)
print("y_test.shape = ", y_test .shape)

from sklearn.metrics import accuracy_score

preds = clf.predict(X_test)
print("Accuracy:", accuracy_score(y_test,preds))

print("took", time.time()-start_time, "seconds")

print("bkpt here to end in debugger")