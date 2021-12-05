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
# explicitly state headers for all six possibilities
all_labels = pd.read_csv("data/train_labels_final_single.csv",names=["image_name", "col1", "col2", "col3", "col4", "col5", "col6"], delimiter=",")
print("two header rows", all_labels.head(5))

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
    print(row[:])
    print(each_row)
    all_rows.append(each_row)
print(all_rows[1])
# pop the row which is not the old header
all_rows.pop(1)



# use this code if you want to do single-label
#Y = np.loadtxt('data/train_labels_final_single.csv', delimiter=',', skiprows=1, usecols=1, dtype=str)

# use this code if you want to do binary Relevance, Label Powerset or Classifier Chain
# create a pandas dataframe of all binarized label data to prep our expected output
# data Y
y = pd.DataFrame(all_rows[1:], columns=all_rows[0])

Y = y[['scab', 'healthy', 'frog_eye_leaf_spot','rust','complex','powdery_mildew']]
Y.to_csv('out.csv')
print("Saved out file")

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



# BinaryRelevance with Logistic Regression Does Not Converge With Any Kind of Setting
# Will need to try and do PCA.

#def logreg():

    #parameters = [
    #{
        #'classifier': [LogisticRegression(multi_class="multinomial", tol=0.015)],
        #'classifier__C': [2.0, 10.0, 100],
        #'classifier__max_iter': [1000]
    #},

    #]

    #clf = GridSearchCV(BinaryRelevance(), parameters, cv = 5, scoring='f1_micro')
    #print("Starting Fit")
    #clf.fit(train_data, train_labels)

    #print (clf.best_params_, clf.best_score_)

#logreg()


def ovr():


    clf_ovr = OneVsRestClassifier(LogisticRegression(penalty='l1', class_weight='balanced', solver="liblinear"), n_jobs=-1)
    clf_ovr.fit(train_data, train_labels)

    predictions = clf_ovr.predict(dev_data)

    print("Accuracy :",metrics.accuracy_score(dev_labels, predictions))
    print("Hamming loss ",metrics.hamming_loss(dev_labels,predictions))

    precision = precision_score(dev_labels, predictions, average='micro')
    recall = recall_score(dev_labels, predictions, average='micro')
    f1 = f1_score(dev_labels, predictions, average='micro')
    print("\nMicro-average quality numbers")
    print("Precision: {:.4f}, Recall: {:.4f}, F1-measure: {:.4f}".format(precision, recall, f1))

    precision = precision_score(dev_labels, predictions, average='macro')
    recall = recall_score(dev_labels, predictions, average='macro')
    f1 = f1_score(dev_labels, predictions, average='macro')
    print("\nMacro-average quality numbers")
    print("Precision: {:.4f}, Recall: {:.4f}, F1-measure: {:.4f}".format(precision, recall, f1))

    print("\nClassification Report")
    print (metrics.classification_report(dev_labels, predictions))


#ovr()

def dim_reduce():

# Do feature scaling. From this the fraction of total variance for the first k
# principle components 25 is 87% and 50 is 1

  sc = StandardScaler()
  x_scaled = sc.fit_transform(X)
  print("Scaled Data")

  # Principal component analysis using scaled train data
  pca = PCA(n_components=50)
  pca.fit(x_scaled)


  # Get the explained variance array. Can also use explained_variance_ratio
  pca_variance = pca.explained_variance_

  # Get fraction of total variance explained by first k principal components
  for k in (25,50):
    print("Fraction of total variance for the first", k, "principal components:",
          sum(pca_variance[0:k])/sum(pca_variance[0:]))
  print("\n\n")

#dim_reduce()

def dim_reduct_transform():

      # scale all the data
      sc = StandardScaler()
      x_scaled = sc.fit_transform(X)

      # Principal component analysis using scaled data
      pca = PCA(n_components=50)
      x_pca = pca.fit_transform(x_scaled)
      print ("PCA transform done")

      # split the pca data
      print("Starting split of train pca and dev pca data")
      train_data_pca,dev_data_pca,train_labels_pca,dev_labels_pca = train_test_split(x_pca,Y,test_size=0.2,random_state=42)

      clf_ovr = OneVsRestClassifier(LogisticRegression(penalty='l1', class_weight='balanced', solver="liblinear"), n_jobs=-1)
      clf_ovr.fit(train_data_pca, train_labels_pca)
      print("OVR fit done")

      predictions = clf_ovr.predict(dev_data_pca)

      print("Accuracy :",metrics.accuracy_score(dev_labels_pca, predictions))
      print("Hamming loss ",metrics.hamming_loss(dev_labels_pca,predictions))

      precision = precision_score(dev_labels_pca, predictions, average='micro')
      recall = recall_score(dev_labels_pca, predictions, average='micro')
      f1 = f1_score(dev_labels_pca, predictions, average='micro')
      print("\nMicro-average quality numbers")
      print("Precision: {:.4f}, Recall: {:.4f}, F1-measure: {:.4f}".format(precision, recall, f1))

      precision = precision_score(dev_labels_pca, predictions, average='macro')
      recall = recall_score(dev_labels_pca, predictions, average='macro')
      f1 = f1_score(dev_labels_pca, predictions, average='macro')
      print("\nMacro-average quality numbers")
      print("Precision: {:.4f}, Recall: {:.4f}, F1-measure: {:.4f}".format(precision, recall, f1))

      print("\nClassification Report")
      print (metrics.classification_report(dev_labels_pca, predictions))

      # Try GridSearchCV, Binary Relevance with Logistic Regression

      parameters = [
        {
              'classifier': [LogisticRegression(multi_class="auto", tol=0.015, solver="liblinear")],
              'classifier__C': [0.01, 0.1, 1.0,2.0, 10.0, 100],
              'classifier__max_iter': [25,50]
         },

      ]

      clf = GridSearchCV(BinaryRelevance(), parameters, cv = 5, scoring='f1_micro')
      print("Starting Fit")
      clf.fit(train_data_pca, train_labels_pca)

      print("Here are the best settings")
      print (clf.best_params_, clf.best_score_)

dim_reduct_transform()
