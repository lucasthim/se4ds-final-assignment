from typing import Callable, List

class Model:
    pass

class Evaluator:
    def __init__(self, model: Model):
        self.model = model

    def evaluate(self, X, y, metrics: List[Callable]):

        y_pred = self.model.predict(X)  # TODO Feature Improvement: some metrics might not work with prediction, but with prediction_proba.
        
        metric_values = {}

        for metric in metrics:
            metric_name = metric.__name__
            metric_values[metric_name] = metric(y_pred, y)
        self.metrics = metric_values

        return self.metrics

    @property
    def metrics(self):
        return self._metrics

    @metrics.setter
    def metrics(self, metrics: dict):
        self._metrics = metrics
