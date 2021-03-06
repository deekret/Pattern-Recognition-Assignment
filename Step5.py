# General
import pandas as pd
import numpy as np
# Statistics
import statistics as st

# Image processing
import matplotlib.pyplot as plt

# Multinomial Logit
from sklearn.linear_model import LogisticRegression

# Support Vector Machine
from sklearn.svm import SVC

# Neural Networks
from sklearn.neural_network import MLPClassifier

# Parameter tuning
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.model_selection import train_test_split

# Load dataset of handwritten digits from URL (takes approx. 30 secs)
import seaborn as sn
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import scale
import cv2
from sklearn.model_selection import cross_validate, cross_val_predict, cross_val_score
from sklearn import metrics
from sklearn import svm

data_set = pd.read_csv("mnist.csv")
mnist_data = data_set.values

# Get useful variables
labels = mnist_data[:, 0]
digits = mnist_data[:, 1:]
img_size = 28

resized_digits = []
for index in range(len(digits)):
    resized_digits.append(cv2.resize(np.array(digits[index], dtype='uint8'), (14,14), interpolation = cv2.INTER_AREA))

X_train, X_test, y_train, y_test = train_test_split(digits, labels, test_size=0.88095)

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

c_values = [0.001, 0.005, 0.01, 0.015, 0.02, 0.03]
model_scores = []
seed = 0

# Fit models with all C values and determine best accuracy and best C value
# Best value is 0.005
"""
for c in c_values:
    log = LogisticRegression(penalty='l1', C=c, solver='liblinear', random_state=seed, max_iter=7600)
    log.fit(X_train, y_train)
    cv_results = cross_validate(log, X_train, y_train, cv=5)
    #log_score = log.score(X_test, y_test)
    model_scores.append(cv_results)
    print("Multinomial logit model with C =", c, "scored", str(cv_results['test_score']))
    print("Average: " + str(sum(cv_results['test_score']) / 5.0))
"""

log = LogisticRegression(penalty='l1', C=0.005, solver='liblinear', random_state=seed, max_iter=7600)
log.fit(X_train, y_train)
log_preds = log.predict(X_test)

log_cm = confusion_matrix(y_test, log_preds)

plt.figure(figsize = (10,7))
sn.heatmap(log_cm, annot=True)
plt.show()

print("R2 score for log model:" + str(log.score(X_test, y_test)))
print("Accuracy score for log model: " + str(metrics.accuracy_score(y_test, log_preds)))
"""
cv_results = cross_validate(log, X_train, y_train, cv=8)
print(cv_results.keys())
print(cv_results['test_score'])

print("Score of model: " + str(log.fit(X_train, y_train).score(X_test, y_test)))
log_preds = log.predict(X_test)

log_cm = confusion_matrix(y_test, log_preds)

plt.figure(figsize = (10,7))
sn.heatmap(log_cm, annot=True)
plt.show()
"""


svm_model = svm.SVC(kernel='poly', C=1, gamma=0.1, random_state=seed, max_iter=7600)
svm_model.fit(X_train, y_train)
cv_results = cross_validate(svm_model, X_train, y_train, cv=5)
model_scores.append(cv_results)
print("SVM model scored", str(cv_results['test_score']))
print("Average: " + str(sum(cv_results['test_score']) / 5.0))

svm_model = svm.SVC(kernel='poly', C=1000000, gamma=1, random_state=seed, max_iter=7600)
svm_model.fit(X_train, y_train)
cv_results = cross_validate(svm_model, X_train, y_train, cv=5)
model_scores.append(cv_results)
print("SVM model scored", str(cv_results['test_score']))
print("Average: " + str(sum(cv_results['test_score']) / 5.0))


model_scores = []
c_values = [ 10**x for x in range(10) ]
g_values = [0.0001, 0.001, 0.01, 0.1, 0.5, 0.7, 0.9, 1.0]
kernel_values = ['linear', 'poly', 'rbf', 'sigmoid']
param_grid = {'C': [0.1,1, 10, 100], 'gamma': [1,0.1,0.01,0.001],'kernel': ['rbf', 'poly', 'sigmoid']}

#grid = GridSearchCV(SVC(),param_grid,refit=True,verbose=2)
#grid.fit(X_train,y_train)

#print(grid.best_estimator_)

svm_model = svm.SVC(kernel='poly', gamma=1, C=0.1, random_state=seed, max_iter=7600)
svm_model.fit(X_train, y_train)
cv_results = cross_validate(svm_model, X_train, y_train, cv=5)
print("SVM model  scored", str(cv_results['test_score']))
print("Average: " + str(sum(cv_results['test_score']) / 5.0))

y_preds = svm_model.predict(X_train)
print("Accuracy: ", svm_model.score(X_test, y_test))

"""
for c in c_values:
    for g in g_values:
        svm_model = svm.SVC(kernel='poly', gamma=g, C=c, random_state=seed, max_iter=7600)
        svm_model.fit(X_train, y_train)
        cv_results = cross_validate(svm_model, X_train, y_train, cv=5)
        model_scores.append(cv_results)
        print("SVM model with C =", c, ",", g, "scored", str(cv_results['test_score']))
        print("Average: " + str(sum(cv_results['test_score']) / 5.0))
"""

#svm_model = svm.SVC(kernel='rbf', gamma=0.1, C=0.001)

#svm_model.fit(X_train, y_train)
#svm_preds = svm_model.predict(X_test)
#print("R2 score for svm model:" + str(svm_model.score(X_test, y_test)))
#print("Accuracy score for svm model: " + str(metrics.accuracy_score(y_test, svm_preds)))