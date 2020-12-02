# Utils
import pandas as pd
import numpy as np
import statistics as st
import cv2
import csv

# Plotting
import matplotlib.pyplot as plt
import seaborn as sn

# Data manipulation
from sklearn.preprocessing import scale
from sklearn.metrics import confusion_matrix

# Models
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

# Parameter tuning
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.model_selection import train_test_split


def print_results(grid_cv, file_name):
    cv_results = grid_cv.cv_results_
    scores_df = pd.DataFrame(cv_results).sort_values(by='rank_test_score')
    scores_df.to_csv(file_name)