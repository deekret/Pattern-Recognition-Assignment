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

log = LogisticRegression(penalty='l1')
log.fit(X_train, y_train)
