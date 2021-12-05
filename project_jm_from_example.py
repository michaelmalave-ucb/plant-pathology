# from https://blog.hyperiondev.com/index.php/2019/02/18/machine-learning/
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.utils import shuffle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

start_time = time.time()

def show_elapsed(msg = ''):
    end_time = time.time()
    global start_time
    seconds = end_time-start_time
    print()
    print("### ", end='')
    if (msg):
        print(msg, 'in ', end='')
    else:
        print('elapsed time = ', end='')
    print(int(seconds//60), ":", int(seconds%60), "(m:s),", int(seconds), " seconds")
    print()
    start_time = time.time()

show_elapsed()

print("### loading...")

# load smaller, train set for this example
train_data = scipy.io.loadmat('./data/train_32x32.mat')

# # load full train set for this example
# train_data = scipy.io.loadmat('./data/extra_32x32.mat')

# extract the images and labels from the dictionary object
X = train_data['X']
y = train_data['y']

print("X.shape = ", X.shape)
print("y.shape = ", y.shape)

show_elapsed('.mat file loaded')

print("### reducing to small X and y for testing...")

small = 5000 #20000
X = X[:,:,:,:small]
y = y[:small,:]

print("X.shape = ", X.shape)
print("y.shape = ", y.shape)

show_elapsed("reduced")

# # view an image (e.g. 25) and print its corresponding label
# img_index = 25
# plt.imshow(X[:,:,:,img_index])
# plt.show()
# print(y[img_index])

print("### flattening...")

X = X.reshape(X.shape[0]*X.shape[1]*X.shape[2],X.shape[3]).T
y = y.reshape(y.shape[0],)
X, y = shuffle(X, y, random_state=42)

print("X.shape = ", X.shape)
print("y.shape = ", y.shape)

show_elapsed('flattened')

print('### clf and split...')

clf = RandomForestClassifier()
print(clf)

RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
           max_depth=None, max_features='auto', max_leaf_nodes=None,
           min_impurity_split=1e-07, min_samples_leaf=1,
           min_samples_split=2, min_weight_fraction_leaf=0.0,
           n_estimators=10, n_jobs=1, oob_score=False, random_state=None,
           verbose=2, warm_start=False)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("X_train.shape = ", X_train.shape)
print("y_train.shape = ", y_train.shape)
print("X_test.shape = ", X_test.shape)
print("y_test.shape = ", y_test .shape)

show_elapsed('clf and split')

print("### fitting...")

clf.fit(X_train, y_train)

show_elapsed('fit')

print("### scoring...")

preds = clf.predict(X_test)
print("Accuracy:", accuracy_score(y_test,preds))

show_elapsed('scored')

pass
