import time
import numpy as np
# import pandas
import pandas as pd
from sklearn import tree
from sklearn import metrics
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report


start = time.time()
print("Starting loading data")

# this method is a little slower than read_csv
#with open("train_data.csv", 'r') as f:
  #X = np.genfromtxt(f, delimiter=',')

# Both pandas.DataFrame and pandas.Series have values attribute that returns
#NumPy array numpy.ndarray. After pandas 0.24.0, it is recommended to use the to_numpy()

X =pd.read_csv("train_data.csv", delimiter=",").values

# handle space delimited labels by splitting label file when there are multiple labels
with open('data/train_labels_final.csv') as infile, open('data/train_labels_final_single.csv', 'w') as outfile:
    for line in infile:
        outfile.write(" ".join(line.split()).replace(' ', ','))
        outfile.write("\n")

# use this code if you want to do single-label
Y = np.loadtxt('data/train_labels_final_single.csv', delimiter=',', skiprows=1, usecols=1, dtype=str)

#Y = np.loadtxt('data/train_labels_final.csv', delimiter=',', skiprows=1, usecols=1, dtype=str)
print("Finished loading data")
end = time.time()
print('Process Complete')
total_time = round(end - start, 2)
print("Total time loading data was: " + str(total_time) + " seconds.")

print('per data shape: ', len(X[0]))
print('data shape: ', X.shape)
print('label shape:', Y.shape)

shuffle = np.random.permutation(np.arange(X.shape[0]))
X, Y = X[shuffle], Y[shuffle]


# Fractionally distribute subsets of data
big_size = round(X.shape[0]*0.8)
small_size = round(X.shape[0]*0.02)

# Set some variables to hold test, dev, and training data.
test_data, test_labels = X[big_size:], Y[big_size:]
dev_data, dev_labels = X[big_size-small_size:big_size], Y[big_size-small_size:big_size]
train_data, train_labels = X[:big_size-small_size], Y[:big_size-small_size]
mini_train_data, mini_train_labels = X[:small_size], Y[:small_size]


# Produce models for each k value
def knn_range(k_values):
  for k in k_values:
    if k == k_values[0]:
      print("Producing models for where k={k_values}".format(k_values=k_values))
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(train_data, train_labels)
    # get predictions for dev data to be evaluated
    pred = knn.predict(dev_data)
    score = knn.score(dev_data, dev_labels)
    print("Accuracy with k={k}: {score}".format(k=k, score=score))
    # Only get classification reports where dev_labels were also in train_labels
    # Insert codehere

    if k == k_values[-1]:
      report = classification_report(pred, dev_labels)
      print("3. Classification report for where k={k}".format(k=k))
      print(report)


k_vals = [1, 3, 5, 7, 9, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 122]
knn_range(k_vals)


# Produce default svc model
# takes long to run
"""clf = SVC(gamma=0.001)
clf.fit(train_data, train_labels)
pred = clf.predict(dev_data)
reg_score = metrics.f1_score(dev_labels, pred, average="weighted")
print("SVC f1-score is: " + str(reg_score))"""


# Produce multinomial naive bayes
clf = MultinomialNB()
clf.fit(train_data, train_labels)
pred = clf.predict(dev_data)
reg_score = metrics.f1_score(dev_labels, pred, average="weighted")
print("Multinomial f1-score is: " + str(reg_score))


# Produce logistic regression model
# takes too long to get result (when past 30 min scaled to 50)
"""clf = LogisticRegression(C=0.01, solver="liblinear", multi_class="auto", max_iter=500)
clf.fit(train_data, train_labels)
pred = clf.predict(dev_data)
reg_score = metrics.f1_score(dev_labels, pred, average="weighted")
print("Logistic regression f1-score is: " + str(reg_score))"""


# Produce decision tree classification
clf = tree.DecisionTreeClassifier()
clf.fit(train_data, train_labels)
pred = clf.predict(dev_data)
reg_score = metrics.f1_score(dev_labels, pred, average="weighted")
print("Decision tree f1-score is: " + str(reg_score))


# Produce default random forest classifier
clf = RandomForestClassifier(max_depth=2, random_state=0)
clf.fit(train_data, train_labels)
pred = clf.predict(dev_data)
reg_score = metrics.f1_score(dev_labels, pred, average="weighted")
print("Decision forest f1-score is: " + str(reg_score))


# Produce default adaboost classifier
clf = AdaBoostClassifier(n_estimators=100, random_state=0)
clf.fit(train_data, train_labels)
pred = clf.predict(dev_data)
reg_score = metrics.f1_score(dev_labels, pred, average="weighted")
print("Adaboost f1-score is: " + str(reg_score))
