from typing import Callable, List, Union
from joblib import dump, load

from numpy.typing import NDArray
from pandas import DataFrame
from sklearn.base import BaseEstimator

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
        path = self._format_path(path)
        self._model = load(path + f"/{model_name}.joblib")

    def save(self,path:str,model_name:str) -> None:
        path = self._format_path(path)
        dump(self._model, path + f"/{model_name}.joblib")

    def _format_path(self,path):
        return path[:-1] if (path[-1] == "/") else path
        
class Evaluator:
    def __init__(self, models: List[Model]):
        self.models = models

    def evaluate(self, X, y, metrics: List[Callable]):

        y_pred = self.model.predict(X) 
        metric_values = {}

        for metric in metrics:
            metric_name = metric.__name__
            metric_values[metric_name] = metric(y_pred, y)
        self._metrics = metric_values

        return self._metrics

    @property
    def get_metrics(self):
        return self._metrics
