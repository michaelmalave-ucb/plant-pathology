# from https://blog.hyperiondev.com/index.php/2019/02/18/machine-learning/
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
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


show_elapsed('startup')

# load full mili data set:
train_data_fname = "train_data.csv"
train_data_labels_final_fname = "data/train_labels_final_single.csv"

# # load 5000 row med mili data set:
# train_data_fname = "train_data_med.csv"
# train_data_labels_final_fname = "data/train_labels_final_single_med.csv"

# # load 1000 row mini mili data set:
# train_data_fname = "train_data_mini.csv"
# train_data_labels_final_fname = "data/train_labels_final_single_mini.csv"

print("### loading", train_data_fname, "...")

# load pp dataset
X =pd.read_csv(train_data_fname, delimiter=",").values

show_elapsed('loaded')

print("### loading", train_data_labels_final_fname, "...")

# get all image names and labels and put into pandas dataframe for further processing
# explicitly state headers for all six possibilities
all_labels = pd.read_csv(train_data_labels_final_fname,names=["image_name", "col1", "col2", "col3", "col4", "col5", "col6"], delimiter=",")
print("note the two header rows:")
print(all_labels.head(5))

show_elapsed('loaded')

print("### building all_rows[] ...")

# empty list where we will turn the label data into binary 0s and 1s to use Binary Relevance
all_rows = []
# a list of all possible labels found in the data
all_possible_labels = ['scab', 'healthy', 'frog_eye_leaf_spot','rust','complex','powdery_mildew']

# append a row to our empty list to represent the new column names
all_rows.append(all_possible_labels)

# turn label data into binary 0 or 1 for each label. 0 signifies that image did not have
# a particular label attached.
for row in all_labels.itertuples(index=False, name='Labels'):
    each_row = []
    if 'scab' in row[:]:
        each_row.append(1)
    else:
        each_row.append(0)
    if 'healthy' in row[:]:
        each_row.append(1)
    else:
        each_row.append(0)
    if 'frog_eye_leaf_spot' in row[:]:
        each_row.append(1)
    else:
        each_row.append(0)
    if 'rust' in row[:]:
        each_row.append(1)
    else:
        each_row.append(0)
    if 'complex' in row[:]:
        each_row.append(1)
    else:
        each_row.append(0)
    if 'powdery_mildew' in row[:]:
        each_row.append(1)
    else:
        each_row.append(0)
    # print(row[:])
    # print(each_row)
    all_rows.append(each_row)
#print(all_rows[1])
# pop the row which is not the old header
all_rows.pop(1)

print("len(all_rows) =", len(all_rows))

show_elapsed('built all_rows')

print("### building X, Y, and y ...")

# use this code if you want to do single-label
#Y = np.loadtxt('data/train_labels_final_single.csv', delimiter=',', skiprows=1, usecols=1, dtype=str)

# use this code if you want to do binary Relevance, Label Powerset or Classifier Chain
# create a pandas dataframe of all binarized label data to prep our expected output
# data Y
y = pd.DataFrame(all_rows[1:], columns=all_rows[0])

Y = y[['scab', 'healthy', 'frog_eye_leaf_spot','rust','complex','powdery_mildew']]

print("X.shape = ", X.shape)
print("Y.shape = ", Y.shape)
print("y.shape = ", y.shape)

show_elapsed('built')

# print("### reducing to small X and y for testing ...")
#
# small = 1000
# X = X[:small,:]
# Y = Y[:small,:]
# y = y[:small,:]
# print("X.shape = ", X.shape)
# print("Y.shape = ", Y.shape)
# print("y.shape = ", y.shape)
#
# show_elapsed('reduced')

# # view an image (e.g. 25) and print its corresponding label
#
# print("### showing one ...")
#
# img_index = 25
# plt.imshow(X[img_index])
# print(y[img_index])
# plt.show()
#
# show_elapsed('shown')

print("### shuffle, build clf, and split ...")

X, y = shuffle(X, y, random_state=42)

clf = RandomForestClassifier()
print(clf)

RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
           max_depth=None, max_features='auto', max_leaf_nodes=None,
           min_impurity_split=1e-07, min_samples_leaf=1,
           min_samples_split=2, min_weight_fraction_leaf=0.0,
           n_estimators=10, n_jobs=1, oob_score=False, random_state=99,
           verbose=2, warm_start=False)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("X_train.shape = ", X_train.shape)
print("y_train.shape = ", y_train.shape)
print("X_test.shape = ", X_test.shape)
print("y_test.shape = ", y_test.shape)

show_elapsed('shuffled, clf, and split')

print("### fitting clf ...")

clf.fit(X_train, y_train)

show_elapsed('fit')

print("### scoring ...")

preds = clf.predict(X_test)
print("Accuracy:", accuracy_score(y_test,preds))

show_elapsed('scored')

pass