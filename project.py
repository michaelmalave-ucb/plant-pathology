import time
import numpy as np
from csv import reader
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report

start = time.time()
print("Starting loading data")
with open("train_data.csv", 'r') as f:
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

# Set some variables to hold test, dev, and training data.
test_data, test_labels = X[big_size:], Y[big_size:]
dev_data, dev_labels = X[big_size-small_size:big_size], Y[big_size-small_size:big_size]
train_data, train_labels = X[:big_size-small_size], Y[:big_size-small_size]
mini_train_data, mini_train_labels = X[:small_size], Y[:small_size]


def q2(k_values):
  for k in k_values:
    # Produce models for each k value
    if k == k_values[0]:
      print("1. Producing models for where k={k_values}".format(k_values=k_values))
      print("2.")
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
q2(k_vals)
