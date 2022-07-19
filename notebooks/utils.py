import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.metrics import f1_score

def format_path(path):
    """Helper function to remove bar from last position of a path."""
    return path[:-1] if (path[-1] == "/") else path

def f1_score_micro(estimator:BaseEstimator, X:pd.DataFrame, y:pd.DataFrame):
    y_pred = estimator.predict(X)
    return f1_score(y_true=y,y_pred=y_pred, average='micro')
