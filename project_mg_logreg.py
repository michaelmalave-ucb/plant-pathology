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
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, hamming_loss, classification_report, precision_score, recall_score,f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.decomposition import PCA, IncrementalPCA

# Just a timer to see load process
start = time.time()
print("Starting loading data into X")


# Both pandas.DataFrame and pandas.Series have values attribute that returns
#NumPy array numpy.ndarray. After pandas 0.24.0, it is recommended to use the to_numpy()
# Here we will read the image fill turned into flattened array into a numpy array

X =pd.read_csv("train_data.csv", delimiter=",").values


# get all image names and labels and put into pandas dataframe for further processing
# explicitly state headers for all six possibilities
all_labels = pd.read_csv("data/train_labels_final_single.csv",names=["image_name", "col1", "col2", "col3", "col4", "col5", "col6"], delimiter=",")


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

# Create an out file to make sure you created the correct binarization for each label.
Y.to_csv('out.csv')
print("Saved out file as a sanity check on binarizing each label")
print("Finished loading data")
end = time.time()
print('Process of Loading X and Y Complete')
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
print("Shapes of data after train, dev split are as follows:")
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


# We will probably need to do PCA for Logistic Regression as it will not Converge
# so lets take the X data and the binarized Y label data and first scale it and then do PCA


# scale all the data
sc = StandardScaler()
x_scaled = sc.fit_transform(X)
print("Scaling of data done")
# Do some sanity check
print(x_scaled.shape)
print(x_scaled[0:10])

# Principal component analysis using scaled data.
# Landed on n_components 60 or 200 after checking for various options and looking at variance capture.
# Note, we experimented with both options
pca = PCA(n_components=200)
x_pca = pca.fit_transform(x_scaled)
print ("PCA transform done")

# split the pca data into train and dev set
print("Starting split of train pca and dev pca data")
train_data_pca,dev_data_pca,train_labels_pca,dev_labels_pca = train_test_split(x_pca,Y,test_size=0.2,random_state=42)

# We tried a One vs Rest classifier in our case. It is similar to Binary Relevance. However this also did not converge
def ovr():

    clf_ovr = OneVsRestClassifier(LogisticRegression(penalty='l1', class_weight='balanced', solver="liblinear"), n_jobs=-1)
    clf_ovr.fit(train_data, train_labels)

    predictions = clf_ovr.predict(dev_data)

    print("Accuracy :",metrics.accuracy_score(dev_labels, predictions))
    print("Hamming loss ",metrics.hamming_loss(dev_labels,predictions))

    precision = precision_score(dev_labels, predictions, average='micro')
    recall = recall_score(dev_labels, predictions, average='micro')
    f1 = f1_score(dev_labels, predictions, average='micro')
    print("\nMicro-average quality numbers for One Vs Rest Logistic Regression")
    print("Precision: {:.4f}, Recall: {:.4f}, F1-measure: {:.4f}".format(precision, recall, f1))

    precision = precision_score(dev_labels, predictions, average='macro')
    recall = recall_score(dev_labels, predictions, average='macro')
    f1 = f1_score(dev_labels, predictions, average='macro')
    print("\nMacro-average quality numbers")
    print("Precision: {:.4f}, Recall: {:.4f}, F1-measure: {:.4f}".format(precision, recall, f1))

    print("\nClassification Report for One Versus Rest Logistic Regression")
    print (metrics.classification_report(dev_labels, predictions))

#ovr()

# Next lets see if we can do some dimenstionality reduction
# Only need to run dim_reduce function once. This is just to help determine
# n_components for PCA reduction
def dim_reduce():


  # Will need to do incremenalPCA because PCA to figure out ideal n_components
  # will not work on this large dataset
  print("Starting Incremental PCA")
  n = x_scaled.shape[0]
  chunk_size = 4000
  ipca = IncrementalPCA(n_components =  2000)
  for i in range(0, n//chunk_size):
    ipca.partial_fit(x_scaled[i*chunk_size : (i+1)*chunk_size])
  print("Ending Incremental PCA")

  # Get the explained variance array. Can also use explained_variance_ratio
  ipca_variance = ipca.explained_variance_

  # Get fraction of total variance explained by first k principal components
  for k in (25,50,60,200,1000):
   print("Fraction of total variance for the first", k, "principal components:",
         sum(ipca_variance[0:k])/sum(ipca_variance[0:]))
  print("\n\n")


  # KEEP THE RESULTS BELOW
  # Results of Incremental PCA used to do dimensionality reduction.
  #Fraction of total variance for the first 25 principal components: 0.5898387666140356
  #Fraction of total variance for the first 50 principal components: 0.6778640348589169
  #Fraction of total variance for the first 60 principal components: 0.7000517524379772
  #Fraction of total variance for the first 200 principal components: 0.8312620486174991
  #Fraction of total variance for the first 1000 principal components: 0.9651329500281319

  # Principal component analysis using scaled train data from up above
  #pca_test = PCA()
  #pca_test.fit(x_scaled)

  # Get the explained variance array. Can also use explained_variance_ratio
  #pca_variance = pca_test.explained_variance_

  # Get fraction of total variance explained by first k principal components
  #for k in (25,50,60,200,1000):
    #print("Fraction of total variance for the first", k, "principal components:",
          #sum(pca_variance[0:k])/sum(pca_variance[0:]))
  #print("\n\n")

  #Do a sanity check on the variance capture calculations
  #print(pca_test.explained_variance_ratio_.cumsum())

#dim_reduce()

# Now try One Versus Rest Logistic Regression Using PCA'd data and we will Grid
# Search for best parameters to use with Binary Relevance Logistic Regression, Label
# Powerset Logistic Regression and Class Chains Logistic Regression.

def dim_reduct_transform():


      clf_ovr = OneVsRestClassifier(LogisticRegression(penalty='l1', class_weight='balanced', solver="liblinear"), n_jobs=-1)
      clf_ovr.fit(train_data_pca, train_labels_pca)
      print("OVR Logistic Regression PCA'd data fit done")

      predictions = clf_ovr.predict(dev_data_pca)

      print("Accuracy for OVR Logistic Regression PCA'd data :",metrics.accuracy_score(dev_labels_pca, predictions))
      print("Hamming loss for OVR Logistic Regression PCA'd data: ",metrics.hamming_loss(dev_labels_pca,predictions))

      precision = precision_score(dev_labels_pca, predictions, average='micro')
      recall = recall_score(dev_labels_pca, predictions, average='micro')
      f1 = f1_score(dev_labels_pca, predictions, average='micro')
      print("\nMicro-average quality numbers for OVR Logistic Regression PCA'd data:")
      print("Precision: {:.4f}, Recall: {:.4f}, F1-measure: {:.4f}".format(precision, recall, f1))

      precision_mac = precision_score(dev_labels_pca, predictions, average='macro')
      recall_mac = recall_score(dev_labels_pca, predictions, average='macro')
      f1_mac = f1_score(dev_labels_pca, predictions, average='macro')
      print("\nMacro-average quality numbers for OVR Logistic Regression PCA'd data")
      print("Precision: {:.4f}, Recall: {:.4f}, F1-measure: {:.4f}".format(precision_mac, recall_mac, f1_mac))

      print("\nClassification Report for OVR Logistic Regression PCA'd data:")
      print (metrics.classification_report(dev_labels_pca, predictions))

      # Try GridSearchCV, Binary Relevance with Logistic Regression

      print("Moving to Grid Search CV for Logistic Regression for PCA'd data")
      parameters = [
        {
              'classifier': [LogisticRegression(multi_class="auto", tol=0.015, solver="liblinear")],
              'classifier__C': [1.0e-10, 0.0001, 0.001, 0.01, 0.1, 0.5, 1.0, 2.0, 10.0, 100],
              'classifier__max_iter': [25,50]
         },

      ]

      # Grid Search Binary Relevance Logistic Regression
      clf = GridSearchCV(BinaryRelevance(), parameters, cv = 5, scoring='f1_micro')
      print("Starting Fit of Grid Search for Binary Relevance")
      clf.fit(train_data_pca, train_labels_pca)

      print("Here are the best settings for Binary Relevance")
      print (clf.best_params_, clf.best_score_)

      # Try GridSearchCV, Label Powerset with Logistic Regression

      clf_labelpower = GridSearchCV(LabelPowerset(), parameters, cv = 5, scoring='f1_micro')
      print("Starting Fit of Grid Search for Label Powerset")
      clf_labelpower.fit(train_data_pca, train_labels_pca)

      print("Here are the best settings for Label Power Set")
      print (clf_labelpower.best_params_, clf_labelpower.best_score_)

      # Try GridSearchCV, Chain Classifier with Logistic Regression
      clf_chain = GridSearchCV(ClassifierChain(), parameters, cv = 5, scoring='f1_micro')
      print("Starting Fit of Grid Search For Classifer Chain")
      clf_chain.fit(train_data_pca, train_labels_pca)

      print("Here are the best settings for Classifier Chains")
      print (clf_chain.best_params_, clf_chain.best_score_)



dim_reduct_transform()

# Now that we have found the best parameters, we will do Logistic Regression with Binary BinaryRelevance
# Label Powerset and Classifer Chain
def bin_logreg():

    # Create classifer for Binary Relevance Logistic Regression Using PCA'd data
    logreg_bin = BinaryRelevance(LogisticRegression(C=.0001,solver="liblinear", multi_class="auto", max_iter=25, tol=0.015))
    print("Created classifier for Binary Relevance Logistic Regression")

    logreg_bin.fit(train_data_pca, train_labels_pca)
    print("Fit the classifier for Binary Relevance Logistic Regression")

    # get predictions for dev data to be evaluated
    pred_bin_log = logreg_bin.predict(dev_data_pca)
    print("Predicted the model for Binary Relevance Logistic Regression")

    # get the accuracy scores
    score = accuracy_score(dev_labels_pca, pred_bin_log)
    print("Accuracy with Binary Relevance Logistic Regression and C={c}: {score}".format(c=.0001, score=score))

    # hamming
    ham = hamming_loss(dev_labels_pca, pred_bin_log)
    print("Hamming score with Binary Relevance Logistic Regression with C={c}: {ham_score}".format(c=.0001, ham_score=ham))

    precision = precision_score(dev_labels_pca, pred_bin_log, average='micro')
    recall = recall_score(dev_labels_pca, pred_bin_log, average='micro')
    f1 = f1_score(dev_labels_pca, pred_bin_log, average='micro')
    print("\nMicro-average quality numbers for Binary Relevance and Logistice Regression Using PCA'd data:")
    print("Precision: {:.4f}, Recall: {:.4f}, F1-measure: {:.4f}".format(precision, recall, f1))

bin_logreg()


def power_logreg():

    # Create classifer for Label Powerset Logistic Regression Using PCA'd data
    logreg_power = LabelPowerset(LogisticRegression(C=2.0,solver="liblinear", multi_class="auto", max_iter=25, tol=0.015))
    print("Created classifier for Label Powerset Logistic Regression")

    logreg_power.fit(train_data_pca, train_labels_pca)
    print("Fit the classifier for Label Powerset Logistic Regression")

    # get predictions for dev data to be evaluated
    pred_power = logreg_power.predict(dev_data_pca)
    print("Predicted the model for Label Powerset Logistic Regression")

    # get the accuracy scores
    score_power = accuracy_score(dev_labels_pca, pred_power)
    print("Accuracy with Label Powerset Logistic Regression and C={c}: {score}".format(c=2.0, score=score_power))

    # hamming
    ham_power = hamming_loss(dev_labels_pca, pred_power)
    print("Hamming score with Label Powerset Logistic Regression with C={c}: {ham_score}".format(c=2.0, ham_score=ham_power))

    precision_power = precision_score(dev_labels_pca, pred_power, average='micro')
    recall_power = recall_score(dev_labels_pca, pred_power, average='micro')
    f1_power = f1_score(dev_labels_pca, pred_power, average='micro')
    print("\nMicro-average quality numbers for Label Powerset and Logistice Regression Using PCA'd data:")
    print("Precision: {:.4f}, Recall: {:.4f}, F1-measure: {:.4f}".format(precision_power, recall_power, f1_power))

power_logreg()


def class_logreg():

    # Create classifer for Classifier Chain Logistic Regression Using PCA'd data
    logreg_chain = ClassifierChain(LogisticRegression(C=0.0001,solver="liblinear", multi_class="auto", max_iter=25, tol=0.015))
    print("Created classifier for Classifer Chain Logistic Regression")

    logreg_chain.fit(train_data_pca, train_labels_pca)
    print("Fit the classifier for Classifier Chain Logistic Regression")

    # get predictions for dev data to be evaluated
    pred_chain = logreg_chain.predict(dev_data_pca)
    print("Predicted the model for Classifier Chain Logistic Regression")

    # get the accuracy scores
    score_chain = accuracy_score(dev_labels_pca, pred_chain)
    print("Accuracy with Classifier Chain Logistic Regression and C={c}: {score}".format(c=0.0001, score=score_chain))

    # hamming
    ham_chain = hamming_loss(dev_labels_pca, pred_chain)
    print("Hamming score with Classifier Chain Logistic Regression with C={c}: {ham_score}".format(c=0.0001, ham_score=ham_chain))

    precision_chain = precision_score(dev_labels_pca, pred_chain, average='micro')
    recall_chain = recall_score(dev_labels_pca, pred_chain, average='micro')
    f1_chain = f1_score(dev_labels_pca, pred_chain, average='micro')
    print("\nMicro-average quality numbers for Classifier Chain and Logistice Regression Using PCA'd data:")
    print("Precision: {:.4f}, Recall: {:.4f}, F1-measure: {:.4f}".format(precision_chain, recall_chain, f1_chain))

class_logreg()
