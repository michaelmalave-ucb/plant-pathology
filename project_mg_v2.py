import time
import numpy as np
import pandas as pd
from skmultilearn.problem_transform import BinaryRelevance
from skmultilearn.problem_transform import LabelPowerset
from skmultilearn.problem_transform import ClassifierChain
from sklearn import tree
from sklearn import metrics
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, hamming_loss, classification_report
from sklearn.model_selection import train_test_split


start = time.time()
print("Starting loading data")

# this method is a little slower than read_csv so we can remove
#with open("train_data.csv", 'r') as f:
  #X = np.genfromtxt(f, delimiter=',')

# Both pandas.DataFrame and pandas.Series have values attribute that returns
#NumPy array numpy.ndarray. After pandas 0.24.0, it is recommended to use the to_numpy()
# Here we will read the image fill turned into flattened array into a numpy array

X =pd.read_csv("train_data.csv", delimiter=",").values


# get all image names and labels and put into pandas dataframe for further processing
all_labels = pd.read_csv("data/train_labels_final_single.csv", delimiter=",")

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
    all_rows.append(each_row)



# use this code if you want to do single-label
#Y = np.loadtxt('data/train_labels_final_single.csv', delimiter=',', skiprows=1, usecols=1, dtype=str)

# use this code if you want to do binary Relevance, Label Powerset or Classifier Chain
# create a pandas dataframe of all binarized label data to prep our expected output
# data Y
y = pd.DataFrame(all_rows[1:], columns=all_rows[0])
Y = y[['scab', 'healthy', 'frog_eye_leaf_spot','rust','complex','powdery_mildew']]

print("Finished loading data")
end = time.time()
print('Process Complete')
total_time = round(end - start, 2)
print("Total time loading data was: " + str(total_time) + " seconds.")

# some sanity check to see the shape of both Y and X before we split for train and dev
print('per data shape: ', len(X[0]))
print('data shape: ', X.shape)
print('label shape:', Y.shape)


# set some variables to train and dev data
print("Starting split of train and dev data")
train_data,dev_data,train_labels,dev_labels = train_test_split(X,Y,test_size=0.2,random_state=42)

# some further sanity checks to make sure we have the right shapes after we do the split
print("Train data shape", train_data.shape)
print("Train label shape", train_labels.shape)
print("Dev data shape", dev_data.shape)
print("Dev label shape", dev_labels.shape)

print("Ending split of train and dev data")




# Produce a KNN model with Problem Transformation using Binary Relevance
def knn_range_binary(k_values):
    for k in k_values:
        if k == k_values[0]:
            print("Producing models for Binary Relevance where k={k_values}".format(k_values=k_values))
        knn_bin = BinaryRelevance(KNeighborsClassifier(n_neighbors=k))
        print("Created classifier for Binary Relevance")
        knn_bin.fit(train_data, train_labels)
        print("Fit the classifier for Binary Relevance")
        # get predictions for dev data to be evaluated
        pred_bin = knn_bin.predict(dev_data)
        print("Predicted the model for Binary Relevance")

        # get the accuracy scores
        score = accuracy_score(dev_labels, pred_bin)
        print("Accuracy with Binary Relevance and k={k}: {score}".format(k=k, score=score))

        # hamming
        ham = hamming_loss(dev_labels, pred_bin)
        print("Hamming score with Binary Relevance with k={k}: {ham_score}".format(k=k, ham_score=ham))


# rule of thumb for k value is square root of number of samples so 112
k_vals = [1, 3, 5, 7, 9, 20, 30, 40, 50, 60, 70, 80, 90, 100, 112, 120, 122]
knn_range_binary(k_vals)

# Produce a KNN model with Problem Transformation using Label Powerset
def knn_range_powerset(k_values):
    for k in k_values:
        if k == k_values[0]:
            print("Producing models for Label Powerset where k={k_values}".format(k_values=k_values))
        knn_powerset = LabelPowerset(KNeighborsClassifier(n_neighbors=k))
        print("Created classifier for Label Powerset")
        knn_powerset.fit(train_data, train_labels)
        print("Fit the classifier for Label Powerset")
        # get predictions for dev data to be evaluated
        pred_powerset = knn_powerset.predict(dev_data)
        print("Predicted the model for Label Powerset")

        # get the accuracy scores
        score_powerset = accuracy_score(dev_labels, pred_powerset)
        print("Accuracy with Label Powerset and k={k}: {score}".format(k=k, score=score_powerset))

        # hamming
        ham_powerset = hamming_loss(dev_labels, pred_powerset)
        print("Hamming score with Label Powerset with k={k}: {ham_score}".format(k=k, ham_score=ham_powerset))


# rule of thumb for k value is square root of number of samples so 112
k_vals = [1, 3, 5, 7, 9, 20, 30, 40, 50, 60, 70, 80, 90, 100, 112, 120, 122]
knn_range_powerset(k_vals)

# Produce a KNN model with Problem Transformation using Classifier Chains
def knn_range_chain(k_values):
    for k in k_values:
        if k == k_values[0]:
            print("Producing models for Classifier Chains where k={k_values}".format(k_values=k_values))
        knn_chains = ClassifierChain(KNeighborsClassifier(n_neighbors=k))
        print("Created classifier for Classifier Chains")
        knn_chains.fit(train_data, train_labels)
        print("Fit the classifier for Classifier Chains")
        # get predictions for dev data to be evaluated
        pred_chains = knn_chains.predict(dev_data)
        print("Predicted the model for Classifier Chains")

        # get the accuracy scores
        score_chains = accuracy_score(dev_labels, pred_chains)
        print("Accuracy with Classifier Chains and k={k}: {score}".format(k=k, score=score_chains))

        # hamming
        ham_chain = hamming_loss(dev_labels, pred_chains)
        print("Hamming score with Classifier Chains with k={k}: {ham_score}".format(k=k, ham_score=ham_chain))


# rule of thumb for k value is square root of number of samples so 112
k_vals = [1, 3, 5, 7, 9, 20, 30, 40, 50, 60, 70, 80, 90, 100, 112, 120, 122]
knn_range_chain(k_vals)
