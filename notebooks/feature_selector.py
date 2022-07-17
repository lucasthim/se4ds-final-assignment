import pandas as pd

from sklearn.feature_selection import RFECV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.base import BaseEstimator

from sklearn.metrics import f1_score



class FeatureSelector():
    """Class that contains the method select specific features that can increase models performance."""

    def __init__(self,random_state=42) -> None:
        self.random_state = random_state

    def select_best_features(self,X:pd.DataFrame,y:pd.DataFrame):

        def f1_score_micro(estimator:BaseEstimator,X:pd.DataFrame,y:pd.DataFrame):
            y_pred = estimator.predict(X)
            return f1_score(y_true=y,y_pred=y_pred, average='micro')

        selector = RFECV(estimator=RandomForestClassifier(min_samples_split=3, min_samples_leaf=3,random_state=self.random_state),
                                    step=1,
                                    min_features_to_select=3,
                                    cv = StratifiedKFold(n_splits=5,random_state=self.random_state,shuffle=True),
                                    scoring=f1_score_micro,
                                    verbose=0,
                                    n_jobs=-1)
        selector.fit(X,y)
        self._selected_features = selector.get_feature_names_out()
        print("Atributos Selecionados:",selector.get_feature_names_out())
    
    @property
    def get_selected_features(self):
        return self._selected_features
        
# TODO: feature selection ou feature extraction.