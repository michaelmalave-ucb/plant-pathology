import time
import numpy as np
from sklearn import tree
from sklearn import metrics
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

prog_start_time = time.time()

start = time.time()
print("Starting loading data")
with open("data/train_data.csv", 'r') as f:
  X = np.genfromtxt(f, delimiter=',')
Y = np.loadtxt('data/train_labels.csv', delimiter=',', skiprows=1, usecols=1, dtype=str)
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

# Normalize pixel values to be between 0 and 1
X = X/225.0

# Set some variables to hold test, dev, and training data.
test_data, test_labels = X[big_size:], Y[big_size:]
dev_data, dev_labels = X[big_size-small_size:big_size], Y[big_size-small_size:big_size]
train_data, train_labels = X[:big_size-small_size], Y[:big_size-small_size]


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
    if k == k_values[-1]:
      report = classification_report(pred, dev_labels)
      print("3. Classification report for where k={k}".format(k=k))
      print(report)


k_vals = [1, 3, 5, 7, 9]
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
# low baseline and might not be worth time spent
"""clf = tree.DecisionTreeClassifier()
clf.fit(train_data, train_labels)
pred = clf.predict(dev_data)
reg_score = metrics.f1_score(dev_labels, pred, average="weighted")
print("Decision tree f1-score is: " + str(reg_score))"""


# Produce default random forest classifier
# low baseline and might not be worth time spent
"""clf = RandomForestClassifier(max_depth=2, random_state=0)
clf.fit(train_data, train_labels)
pred = clf.predict(dev_data)
reg_score = metrics.f1_score(dev_labels, pred, average="weighted")
print("Decision forest f1-score is: " + str(reg_score))"""

# Produce default adaboost classifier
clf = AdaBoostClassifier(n_estimators=100, random_state=0)
clf.fit(train_data, train_labels)
pred = clf.predict(dev_data)
reg_score = metrics.f1_score(dev_labels, pred, average="weighted")
print("Adaboost f1-score is: " + str(reg_score))

prog_end_time = time.time()
print('All Models Complete')
prod_total_time = round(end - start, 2)
print("Total time loading data was: " + str(prod_total_time) + " seconds.")
