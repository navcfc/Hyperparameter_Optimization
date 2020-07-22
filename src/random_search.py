import numpy as np
import pandas as pd

from sklearn import ensemble
from sklearn import metrics
from sklearn import model_selection


if __name__=="__main__":
    df = pd.read_csv("../input/train.csv")
    X = df.drop("price_range", axis = 1).values
    y = df.price_range.values

    classifier = ensemble.RandomForestClassifier(n_jobs=-1)
    param_grid = {
        # "n_estimators" : [100,200,300,400],
        # "max_depth" : [1,3,5,7],
        # "criterion" : ["gini", "entropy"]
        "n_estimators" : np.arange(100,1500,100),
        "max_depth" : np.arange(1,15,1),
        "criterion" : ["gini", "entropy"]
    }

    model = model_selection.RandomizedSearchCV(
        estimator=classifier,
        param_distributions= param_grid,
        scoring= "accuracy",
        n_iter=10,
        verbose=10,
        n_jobs=1,
        cv = 5
    )

    model.fit(X, y)
    print(model.best_score_)
    print(model.best_estimator_.get_params())
