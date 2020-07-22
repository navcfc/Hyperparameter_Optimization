import numpy as np
import pandas as pd

from sklearn import ensemble
from sklearn import metrics
from sklearn import model_selection
from sklearn import pipeline
from sklearn import decomposition
from sklearn import preprocessing


if __name__=="__main__":
    df = pd.read_csv("../input/train.csv")
    X = df.drop("price_range", axis = 1).values
    y = df.price_range.values

    scl = preprocessing.StandardScaler()
    pca = decomposition.PCA()
    rf = ensemble.RandomForestClassifier(n_jobs=-1)
    
    classifier = pipeline.Pipeline([
        ("scaling", scl),
        ("pca", pca),
        ("rf", rf)

    ])
    param_grid = {
        # "n_estimators" : [100,200,300,400],
        # "max_depth" : [1,3,5,7],
        # "criterion" : ["gini", "entropy"]
        "pca__n_components" : np.arange(5, 10),
        "rf__n_estimators" : [100,200],
        "rf__max_depth" : [1,3],
        # "criterion" : ["gini", "entropy"]
    }

    model = model_selection.GridSearchCV(
        estimator=classifier,
        param_grid = param_grid,
        scoring= "accuracy",
        verbose=10,
        n_jobs=1,
        cv = 5
    )

    model.fit(X, y)
    print(model.best_score_)
    print(model.best_estimator_.get_params())
