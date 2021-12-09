# Import all necessary packages and modules

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
from sklearn.metrics import accuracy_score, hamming_loss, classification_report, precision_score, recall_score,f1_score
from sklearn.model_selection import train_test_split


# Just a timer to see load process
start = time.time()
print("Starting loading data")


# Both pandas.DataFrame and pandas.Series have values attribute that returns
#NumPy array numpy.ndarray. After pandas 0.24.0, it is recommended to use the to_numpy()
# Here we will read the image fill turned into flattened array into a numpy array

resize_key = '71_100_67_BL'

source_dir = 'data/' + resize_key

train_data_filename = source_dir + '/train_data.csv'
#X =pd.read_csv("train_data.csv", delimiter=",", header=None).values
X =pd.read_csv(train_data_filename, delimiter=",", header=None).values

# get all image names and labels and put into pandas dataframe for further processing
# explicitly state headers for all six possibilities
train_labels_filename = source_dir + '/train_labels_final_single.csv'
#all_labels = pd.read_csv("data/train_labels_final_single.csv",names=["image_name", "col1", "col2", "col3", "col4", "col5", "col6"], delimiter=",")
all_labels = pd.read_csv(train_labels_filename, names=["image_name", "col1", "col2", "col3", "col4", "col5", "col6"], delimiter=",")


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

# pop the row which is not the old header
all_rows.pop(1)


# use this code if you want to do binary Relevance, Label Powerset or Classifier Chain
# create a pandas dataframe of all binarized label data to prep our expected output
# data Y
y = pd.DataFrame(all_rows[1:], columns=all_rows[0])

Y = y[['scab', 'healthy', 'frog_eye_leaf_spot','rust','complex','powdery_mildew']]
Y.to_csv('out.csv')

# Print a file to do a sanity check to make sure lables were binarized correctly
print("Saved sanity check out file to see all labels set to 0,1")

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

# Lets see the category distribution in the train and dev sets

print("The number of samples in the train data for each category are as follows:")


print("Train data scab counts", (train_labels.scab == 1).sum())
print("Train data healthy counts", (train_labels.healthy == 1).sum())
print("Train data frog_eye_leaf_spot counts", (train_labels.frog_eye_leaf_spot == 1).sum())
print("Train data rust counts", (train_labels.rust == 1).sum())
print("Train data complex counts", (train_labels.complex == 1).sum())
print("Train data powdery_mildew counts", (train_labels.powdery_mildew == 1).sum())

print("The number of samples  in the dev data for each category are as follows:")
print("Dev data scab counts", (dev_labels.scab == 1).sum())
print("Dev data healthy counts", (dev_labels.healthy == 1).sum())
print("Dev data frog_eye_leaf_spot counts", (dev_labels.frog_eye_leaf_spot == 1).sum())
print("Dev data rust counts", (dev_labels.rust == 1).sum())
print("Dev data complex counts", (dev_labels.complex == 1).sum())
print("Dev data powdery_mildew counts", (dev_labels.powdery_mildew == 1).sum())


# Result of split. We tested up to k of 119 for KNN model due to 119 being
# square root of 14163

# Train data shape (14163, 20100)
# Train label shape (14163, 6)
# Dev data shape (3541, 20100)
# Dev label shape (3541, 6)

# Produce a KNN model with Problem Transformation using Binary Relevance
def knn_range_binary(k_values):
    for k in k_values:
        if k == k_values[0]:
            print("Producing models for Binary Relevance where k={k_values}".format(k_values=k_values))
        knn_bin = BinaryRelevance(KNeighborsClassifier(n_neighbors=k))
        print("Created classifier for Binary Relevance / KNN")
        knn_bin.fit(train_data, train_labels)
        print("Fit the classifier for Binary Relevance /KNN")
        # get predictions for dev data to be evaluated
        pred_bin = knn_bin.predict(dev_data)
        print("Predicted the model for Binary Relevance/KNN")

        # get the accuracy scores and hamming
        score = accuracy_score(dev_labels, pred_bin)
        print("Accuracy with Binary Relevance and KNN with k={k}: {score}".format(k=k, score=score))


        # hamming
        ham = hamming_loss(dev_labels, pred_bin)
        print("Hamming score with Binary Relevance and KNN with k={k}: {ham_score}".format(k=k, ham_score=ham))


        # Getting F1 micro score is better when categories not evenly distributed
        precision = precision_score(dev_labels, pred_bin, average='micro')
        recall = recall_score(dev_labels, pred_bin, average='micro')
        f1 = f1_score(dev_labels, pred_bin, average='micro')
        print("\nMicro-average quality numbers for Binary Relevance  and KNN with k= ", k)
        print("Precision: {:.4f}, Recall: {:.4f}, F1-measure: {:.4f}".format(precision, recall, f1))


# Tested multiple k values (see below) and then chose to display ones with highest potential
k_vals = [1, 3, 5, 7, 9]

# rule of thumb for k value is square root of number of samples so 112
#k_vals = [1, 3, 5, 7, 9, 20, 30, 40, 50, 60, 70, 80, 90, 100, 112, 120, 122]
knn_range_binary(k_vals)

# Produce a KNN model with Problem Transformation using Label Powerset
def knn_range_powerset(k_values):
    for k in k_values:
        if k == k_values[0]:
            print("Producing models for Label Powerset and KNN where k={k_values}".format(k_values=k_values))
        knn_powerset = LabelPowerset(KNeighborsClassifier(n_neighbors=k))
        print("Created classifier for Label Powerset/KNN")
        knn_powerset.fit(train_data, train_labels)
        print("Fit the classifier for Label Powerset/KNN")
        # get predictions for dev data to be evaluated
        pred_powerset = knn_powerset.predict(dev_data)
        print("Predicted the model for Label Powerset/KNN")

        # get the accuracy scores
        score_powerset = accuracy_score(dev_labels, pred_powerset)
        print("Accuracy with Label Powerset and KNN with k={k}: {score}".format(k=k, score=score_powerset))


        # hamming
        ham_powerset = hamming_loss(dev_labels, pred_powerset)
        print("Hamming score with Label Powerset and KNN with k={k}: {ham_score}".format(k=k, ham_score=ham_powerset))


        # Getting F1 micro score is better when categories not evenly distributed
        precision_power = precision_score(dev_labels, pred_powerset, average='micro')
        recall_power = recall_score(dev_labels, pred_powerset, average='micro')
        f1_power = f1_score(dev_labels, pred_powerset, average='micro')
        print("\nMicro-average quality numbers for Label Powerset and KNN with k= ", k)
        print("Precision: {:.4f}, Recall: {:.4f}, F1-measure: {:.4f}".format(precision_power, recall_power, f1_power))


# rule of thumb for k value is square root of number of samples so 112
#k_vals = [1, 3, 5, 7, 9, 20, 30, 40, 50, 60, 70, 80, 90, 100, 112, 120, 122]

# Tested multiple k values (see above) and then chose to display ones with highest potential
k_vals = [1, 3, 5, 7, 9]
#knn_range_powerset(k_vals)

# Produce a KNN model with Problem Transformation using Classifier Chains
def knn_range_chain(k_values):
    for k in k_values:
        if k == k_values[0]:
            print("Producing models for Classifier Chains and KNN where k={k_values}".format(k_values=k_values))
        knn_chains = ClassifierChain(KNeighborsClassifier(n_neighbors=k))
        print("Created classifier for Classifier Chains/KNN")
        knn_chains.fit(train_data, train_labels)
        print("Fit the classifier for Classifier Chains/KNN")
        # get predictions for dev data to be evaluated
        pred_chains = knn_chains.predict(dev_data)
        print("Predicted the model for Classifier Chains/KNN")

        # get the accuracy scores
        score_chains = accuracy_score(dev_labels, pred_chains)
        print("Accuracy with Classifier Chains and KNN with k={k}: {score}".format(k=k, score=score_chains))


        # hamming
        ham_chain = hamming_loss(dev_labels, pred_chains)
        print("Hamming score with Classifier Chains and KNN with k={k}: {ham_score}".format(k=k, ham_score=ham_chain))

        # Getting F1 micro score is better when categories not evenly distributed
        precision_chain = precision_score(dev_labels, pred_chains, average='micro')
        recall_chain = recall_score(dev_labels, pred_chains, average='micro')
        f1_chain = f1_score(dev_labels, pred_chains, average='micro')
        print("\nMicro-average quality numbers for Chain Classifier and KNN with k= ", k)
        print("Precision: {:.4f}, Recall: {:.4f}, F1-measure: {:.4f}".format(precision_chain, recall_chain, f1_chain))


# rule of thumb for k value is square root of number of samples so 112
#k_vals = [1, 3, 5, 7, 9, 20, 30, 40, 50, 60, 70, 80, 90, 100, 112, 120, 122]

# Tested multiple k values (see above) and then chose to display ones with highest potential
k_vals = [1, 3, 5, 7, 9]
#knn_range_chain(k_vals)
