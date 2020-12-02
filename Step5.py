from imports import *

# Load dataset
data_set = pd.read_csv("mnist.csv")
mnist_data = data_set.values

# Get useful variables
labels = mnist_data[:, 0]
digits = mnist_data[:, 1:]
img_size = 28

# Reduce image
resized_digits = []
for index in range(len(digits)):
    resized_digits.append(cv2.resize(np.array(digits[index], dtype='uint8'), (14, 14), interpolation=cv2.INTER_AREA))

# Split training and test set
X_train, X_test, y_train, y_test = train_test_split(digits, labels, stratify=labels, test_size=0.88095)

print("Shape of training dataset: ", X_train.shape, "   | shape of test dataset: ", X_test.shape)
print("Shape of training labels: ", y_train.shape, "    | shape of test dataset: ", y_test.shape)

# Normalize data
train_mean = X_train.mean()
train_std = X_train.std()


def norm(x):
    return (x - train_mean) / train_std


X_train = norm(X_train)
X_test = norm(X_test)

# Util variables
seed = 0
k = 5
num_thread = 8
max_iter = 500
verbose = 2

### MULTINOMIAL LOGIT MODEL ###
# Define hyperparameters
log_c_values = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100]
log_params = {'C': log_c_values}

# Define Grid Search
log = LogisticRegression(penalty='l1', solver='liblinear', max_iter=max_iter, random_state=seed)
log_cv = GridSearchCV(log, log_params, cv=k, verbose=verbose, n_jobs=num_thread)
log_cv.fit(X_train, y_train)

print_results(log_cv, 'log_results.csv')

print("Best LOG estimator:", log_cv.best_estimator_)
print("Best LOG score:", log_cv.best_score_)
print("Best LOG params:", log_cv.best_params_)

### SUPPORT VECTOR MACHINE ###
# Define hyperparameters
svm_c_values = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100]
g_values = [0.0001, 0.001, 0.01, 0.1, 0.5, 0.7, 0.9, 1.0]
k_values = ['linear', 'poly', 'rbf', 'sigmoid']
svm_params = {'C': svm_c_values, 'gamma': g_values, 'kernel': k_values}

# Define Grid Search
svm = SVC(max_iter=max_iter, random_state=seed)
svm_cv = GridSearchCV(svm, svm_params, cv=k, verbose=verbose, n_jobs=num_thread)
svm_cv.fit(X_train, y_train)

print_results(svm_cv, 'svm_results.csv')

print("Best SVM estimator:", svm_cv.best_estimator_)
print("Best SVM score:", svm_cv.best_score_)
print("Best SVM params:", svm_cv.best_params_)

### FEED FORWARD NEURAL NETWORK ###
# Define hyperparameters
activations = ['logistic', 'tanh', 'relu']
solvers = ['sgd', 'adam']
lr_inits = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100]
mlp_parameters = {'activation': activations, 'solver': solvers, 'learning_rate_init': lr_inits}

# Define Grid Search
mlp = MLPClassifier(max_iter=max_iter, random_state=seed)
mlp_cv = GridSearchCV(mlp, mlp_parameters, cv=k, verbose=verbose, n_jobs=num_thread)
mlp_cv.fit(X_train, y_train)

print_results(mlp_cv, 'mlp_results.csv')

print("Best MLP estimator:", mlp_cv.best_estimator_)
print("Best MLP score:", mlp_cv.best_score_)
print("Best MLP params:", mlp_cv.best_params_)
