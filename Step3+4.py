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

# Load dataset of handwritten digits from URL (takes approx. 30 secs)
import seaborn as sn
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import scale

data_set = pd.read_csv("mnist.csv")
mnist_data = data_set.values

# Get useful variables
labels = mnist_data[:, 0]
digits = mnist_data[:, 1:]
img_size = 28

labellist = [0] * 10
sum = 0
for x in range(len(labels)):
  labellist[labels[x]] += 1;
  sum += 1

# Count pixels that aren't white along X axis
x_counts = []
for d in digits:
  d = d.reshape(img_size, img_size)
  x_counts.append(np.count_nonzero(d, axis = 0))
x_counts = np.array(x_counts)

# Count pixels that aren't white along Y axis
y_counts = []
for d in digits:
  d = d.reshape(img_size, img_size)
  y_counts.append(np.count_nonzero(d, axis = 1))
y_counts = np.array(y_counts)

# Pixel count on X axis
digit_num = 5
pixels = np.arange(img_size)
xvalue = x_counts[digit_num]
yvalue = y_counts[digit_num]

"""
x_model = LogisticRegression()
x_model.fit(x_counts, labels)
x_preds = x_model.predict(x_counts)

x_cm = confusion_matrix(labels, x_preds)

plt.figure(figsize = (10,7))
sn.heatmap(x_cm, annot=True)
plt.show()

xy_counts = []
for i in range(len(digits)):
  xy_counts.append(np.concatenate((x_counts[i], y_counts[i])))
xy_counts = np.array(xy_counts)

xy_model = LogisticRegression()
xy_model.fit(xy_counts, labels)
xy_preds = xy_model.predict(xy_counts)

xy_cm = confusion_matrix(labels, xy_preds)

plt.figure(figsize = (10,7))
sn.heatmap(xy_cm, annot=True)
plt.show()
"""
# Mean of each column value for each class
pixels = np.arange(img_size)
x_means = []
y_means = []
x_stds = []
y_stds = []

for c in range(10):
  counts_per_label = x_counts[labels == c]
  x_mean = np.mean(counts_per_label, axis=0)
  x_std = np.std(counts_per_label, axis=0)
  x_means.append(x_mean)
  x_stds.append(x_std)

  # Same for y
  counts_per_label = y_counts[labels == c]
  y_mean = np.mean(counts_per_label, axis=0)
  y_std = np.std(counts_per_label, axis=0)
  y_means.append(y_mean)
  y_stds.append(y_std)

x_means = np.array(x_means)
x_stds = np.array(x_stds)
y_means = np.array(y_means)
y_stds = np.array(y_stds)

"""for c in range(10):
  plt.bar(pixels, x_means[c])
  plt.plot(x_stds[c], color='yellow')
  plt.xticks(pixels)
  x_title = 'Non-white pixels along X axis for digit' + str(c)
  plt.title = x_title
  plt.xlabel('Column number')
  plt.ylabel('Number of non-white pixels')
  plt.show()

  plt.bar(pixels, y_means[c])
  plt.plot(y_stds[c], color='yellow')
  plt.xticks(pixels)
  y_title = 'Non-white pixels along Y axis for digit' + str(c)
  plt.title = y_title
  plt.xlabel('Column number')
  plt.ylabel('Number of non-white pixels')
  plt.show()
"""

y_regular = y_counts
for i in range(len(y_counts)):
  y_counts[i] = scale(y_counts[i])


y_model = LogisticRegression(penalty='l2', max_iter=7600)
y_model.fit(y_regular, labels)
y_preds = y_model.predict(y_regular)

y_cm = confusion_matrix(labels, y_preds)

plt.figure(figsize = (10,7))
sn.heatmap(y_cm, annot=True)
plt.show()

print("Score for y_count feature: " + str(y_model.fit(y_counts, labels).score(y_counts, labels)))

# Create ink feature
ink_values = []
for digit in digits:
  ink = 0
  for pixel in digit:
    ink += pixel
  ink_values.append(ink)
ink_scaled = np.array(ink_values).reshape(-1, 1)

# Combine the two features
ink_y = []
for i in range(len(digits)):
  ink_y.append(np.concatenate((ink_scaled[i], y_regular[i])))
ink_y = np.array(ink_y)

print(ink_y.shape)

# Train a new model
ink_y_model = LogisticRegression()
ink_y_model.fit(ink_y, labels)
ink_y_preds = ink_y_model.predict(ink_y)

ink_y_cm = confusion_matrix(labels, ink_y_preds)

plt.figure(figsize = (10,7))
sn.heatmap(ink_y_cm, annot=True)
plt.show()

print("Score for ink_y_count feature: " + str(ink_y_model.fit(y_counts, labels).score(y_counts, labels)))