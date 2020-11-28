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

# Remember to pass the same seed with random_state to all models for reproducible results across multiple calls
seed = 0

### MULTINOMIAL LOGIT MODEL ###
# Define range of C values
c_values = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100]
model_scores = []

# Fit models with all C values and determine best accuracy and best C value
for c in c_values:
    log = LogisticRegression(penalty='l1', C=c, solver='liblinear', random_state=seed)
    log.fit(X_train, y_train)
    log_score = log.score(X_test, y_test)
    model_scores.append(log_score)
    print("Multinomial logit model with C =", c, "scored", str(round(log_score * 100, 2)), "% accuracy.")
best_log_score = max(model_scores)
best_c_value = c_values[model_scores.index(best_log_score)]
print("Best accuracy is ", str(round(best_log_score * 100, 2)), "% achieved with C =", best_c_value)

# Refit model with best values and plot confusion matrix
log = LogisticRegression(penalty='l1', C=best_c_value, solver='liblinear', random_state=seed)
log.fit(X_train, y_train)
log_score = log.score(X_test, y_test)
print("Model score: ", str(round(log_score * 100, 2)))
log_pred = log.predict(X_test)
log_cm = confusion_matrix(y_test, log_pred)
plt.figure(figsize=(10, 7))
sn.heatmap(log_cm, annot=True)
plt.show()

### SUPPORT VECTOR MACHINE ###

### FEED FORWARD NEURAL NETWORK ###
# Define range of hyperparameters
activations = ['logistic', 'tanh', 'relu']
solvers = ['sgd', 'adam']
lr_inits = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100]
model_configs = []

# Fit models with all configuration and determine best one
for act in activations:
    for solv in solvers:
        for lr in lr_inits:
            mlp = MLPClassifier(activation=act, solver=solv, learning_rate_init=lr, max_iter=500, random_state=seed)
            mlp.fit(X_train, y_train)
            mlp_score = mlp.score(X_test, y_test)
            model_config = {"score": mlp_score, "activation": act, "solver": solv, "init_lr": lr}
            model_configs.append(model_config)
            print("Neural network accuracy: ", str(round(mlp_score * 100, 2)), " | activation =", act, ", solver =", solv, ", learning_rate_init =", lr)
best_config = max(model_configs, key=lambda x:x['score'])
print("Best neural network configuration: ", best_config)

# Refit model with best configuration and plot confusion matrix
mlp = MLPClassifier(activation=best_config["activation"], solver=best_config["solver"], learning_rate_init=best_config["init_lr"], max_iter=500, random_state=seed)
mlp.fit(X_train, y_train)
mlp_score = mlp.score(X_test, y_test)
print("Model score: ", str(round(mlp_score * 100, 2)))
mlp_pred = mlp.predict(X_test)
mlp_cm = confusion_matrix(y_test, mlp_pred)
plt.figure(figsize=(10, 7))
sn.heatmap(mlp_cm, annot=True)
plt.show()
