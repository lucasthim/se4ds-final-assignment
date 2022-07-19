import pandas as pd

from sklearn.feature_selection import RFECV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.base import BaseEstimator
from sklearn.metrics import f1_score

import yaml

class FeatureSelector():
    """Class that contains the method select specific features that can increase models performance."""

    def __init__(self,random_state=42) -> None:
        self.random_state = random_state

    def select_best_features(self, X:pd.DataFrame, y:pd.DataFrame) -> None:
        """
        Execute the feature selection step. 
        The method that is being used is the Recursive Feature Elimination with Cross Validation.
        The model that is being used here is the RandomForestClassifier.
                
        Parameters:
        -----------
        - X: input features to train the model.
        - y: target variable to train the model.

        Returns:
        -----------
        None

        """

        def f1_score_micro(estimator:BaseEstimator, X:pd.DataFrame, y:pd.DataFrame):
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
        self._selected_features = selector.get_feature_names_out().tolist()
        print("Atributos Selecionados:",self._selected_features)
    
    def save_best_features(self,path:str) -> None:
        """Save the selected features into a separate YAML file."""

        path = self._format_path(path)
        dict_file = {'best_features' : self._selected_features}
        with open(f'{path}/best_features.yaml', 'w') as file:
            documents = yaml.dump(dict_file, file)

            
    def load_best_features(self,path:str)-> list:
        """Load the selected features into memory from a YAML file."""
        path = self._format_path(path)
        with open(f'{path}/best_features.yaml') as file:
            # The FullLoader parameter handles the conversion from YAML scalar values to Python the dictionary format
            features_file = yaml.load(file, Loader=yaml.FullLoader)
        self._selected_features = features_file["best_features"]
        
    def _format_path(self,path):
        return path[:-1] if (path[-1] == "/") else path

    @property
    def get_selected_features(self) -> list:
        return self._selected_features
    