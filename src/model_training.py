from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV

def train_svm(X_train, y_train):
    svm_param_grid = {
        'C': [0.1, 1, 10, 100],
        'kernel': ['linear', 'rbf', 'poly'],
        'gamma': ['scale', 'auto']
    }
    svm_grid = GridSearchCV(SVC(), svm_param_grid, cv=5, scoring='accuracy')
    svm_grid.fit(X_train, y_train)
    return svm_grid.best_estimator_

def train_knn(X_train, y_train):
    knn_param_grid = {
        'n_neighbors': [3, 5, 7, 10],
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan', 'minkowski']
    }
    knn_grid = GridSearchCV(KNeighborsClassifier(), knn_param_grid, cv=5, scoring='accuracy')
    knn_grid.fit(X_train, y_train)
    return knn_grid.best_estimator_

def train_decision_tree(X_train, y_train):
    dt_param_grid = {
        'criterion': ['gini', 'entropy'],
        'max_depth': [None, 10, 20, 30, 50],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    dt_grid = GridSearchCV(DecisionTreeClassifier(), dt_param_grid, cv=5, scoring='accuracy')
    dt_grid.fit(X_train, y_train)
    return dt_grid.best_estimator_

def train_neural_network(X_train, y_train):
    nn_param_grid = {
        'hidden_layer_sizes': [(50,), (100,), (50, 50)],
        'activation': ['tanh', 'relu'],
        'solver': ['sgd', 'adam'],
        'learning_rate': ['constant', 'adaptive'],
        'max_iter': [1000, 2000, 3000],  # Increase the number of iterations
        'early_stopping': [True]  # Enable early stopping
    }
    nn_grid = GridSearchCV(MLPClassifier(), nn_param_grid, cv=5, scoring='accuracy')
    nn_grid.fit(X_train, y_train)
    return nn_grid.best_estimator_
