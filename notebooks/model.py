from typing import List, Union
from joblib import dump, load

from pandas import DataFrame
from numpy.typing import NDArray
from sklearn.base import BaseEstimator

from utils import f1_score_micro,format_path

class Model:
    """
    Main class to access machine learning models. 
    Only models that follow the Scikit Learn standards are accepted.
    """
    def __init__(self,model:BaseEstimator = None) -> None:
        """ 
        Parameters:
        -----------

        model: Estimator that follows the Scikit Learn standards. Might be loaded empty in case a trained model is loaded from storage.

        """
        self._model = model

    def fit(self,X,y) -> None:
        self._model.fit(X,y)

    def predict(self,X) -> Union[NDArray,DataFrame]:
        return self._model.predict(X)
    
    def predict_proba(self,X) -> Union[NDArray,DataFrame]:
        return self._model.predict_proba(X)

    def load(self,path:str,model_name:str) -> None:
        path = format_path(path)
        self._model = load(path + f"/{model_name}.joblib")

    def save(self,path:str,model_name:str) -> None:
        path = format_path(path)
        dump(self._model, path + f"/{model_name}.joblib")


class ModelOptimizer:
    
    def __init__(self) -> None:
        pass

    def optimize_all_models(self,X:DataFrame,y:DataFrame) -> None:
        pass

    def _optimize_logistic_regression(self) -> None:
        pass

    def _optimize_random_forest(self) -> None:
        pass

    def _optimize_light_gbm(self) -> None:
        pass

    @property
    def get_optimized_models(self):
        return self._best_lr,self._best_rf,self._best_lgbm


class ModelSelector:
    """Selects the top performing model based on the f1-score-micro. This step should be the last one executed during experimentation."""

    def __init__(self, models: List[Model],model_names:List[str]):
        """
        Parameters:
        -----------
        models: List with trained models.
        
        model_names: List with the model names.
        
        """
        self.models = models
        self.model_names = model_names
        self.model_scores = []
        
        self._winner_model = None
        self._winner_score = 0
        self._winner_model_name = ""

    def select_best_model(self,X,y):

        for name,model in zip(self.model_names,self.models):
            test_score = f1_score_micro(estimator = model,X=X,y=y)
            self.model_scores.append(test_score)
            if test_score > self._winner_score:
                self._winner_model = model
                self._winner_score = test_score
                self._winner_model_name = name

            print(f"{name} final score: %.4f" % test_score)

        print(f'\nBest model is {self._winner_model_name} with f1-score-micro = {self._winner_score : .4f} for the test set.')

    @property
    def get_winner_model(self):
        return self._winner_model
