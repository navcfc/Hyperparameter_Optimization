import numpy as np
import pandas as pd

from sklearn import ensemble
from sklearn import metrics
from sklearn import model_selection
from sklearn import pipeline
from sklearn import decomposition
from sklearn import preprocessing

from functools import partial
from skopt import space, gp_minimize
from hyperopt import hp, fmin, tpe, Trials
from hyperopt.pyll.base import scope

import optuna

def optimize(trial, X, y):
    
    criterion = trial.suggest_categorical("criterion", ["gini","entropy"])
    n_estimators = trial.suggest_int("n_estimators", 100,1500)
    max_depth = trial.suggest_int("max_depth",3, 15)
    max_features = trial.suggest_uniform("max_features",0.01, 1.0)

    model = ensemble.RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        max_features=max_features,
        criterion=criterion
    )
    kf  = model_selection.StratifiedKFold(n_splits=5)
    accuracies = []
    for idx in kf.split(X, y):
        train_idx, test_idx = idx[0], idx[1]
        x_train = X[train_idx]
        y_train = y[train_idx]

        x_test = X[test_idx]
        y_test = y[test_idx]
        
        model.fit(x_train,y_train)
        preds = model.predict(x_test)
        fold_acc = metrics.accuracy_score(y_test, preds)
        accuracies.append(fold_acc)

    return -1*np.mean(accuracies)

if __name__=="__main__":
    df = pd.read_csv("../input/train.csv")
    X = df.drop("price_range", axis = 1).values
    y = df.price_range.values

    optimization_function = partial(optimize, X=X, y= y )
    
    study = optuna.create_study(direction="minimize")

    study.optimize(optimization_function, n_trials = 15)

