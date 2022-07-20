from typing import List, Union
from joblib import dump, load

from pandas import DataFrame
from numpy.typing import NDArray
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import StratifiedKFold

from lightgbm import LGBMClassifier

from optuna import distributions
from optuna.integration import OptunaSearchCV

from src.utils import f1_score_micro,format_path

RANDOM_STATE = 42
OPTUNA_N_TRIALS = 200
CV_SPLITS = 5

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
    
    """Class to optimize three pre configured ML models: Logistic Regression, Random Forest and Light GBM"""
    def __init__(self,random_state = RANDOM_STATE, optimization_trials = OPTUNA_N_TRIALS, cv_splits = CV_SPLITS) -> None:
        """
        Parameters
        ----------

        random_state: Random seed to lock experiment random states. This is important to keep reproducibility.

        optimization_trials: Number of trials to run the optimization process. 
        
        cv_splits: Number of cross validation splits to evaluate the model.
        """
        self.random_state = random_state
        self.optimization_trials = optimization_trials
        self.cv_splits = cv_splits
        self._best_lr,self._best_rf,self._best_lgbm = None, None, None

    def optimize_all_models(self,X:DataFrame,y:DataFrame) -> None:
        """
        Run the optimization process for 3 models: Logistic Regression, Random Forest and Light GBM.
        
        Parameters
        ----------
        X: dataset containing features to train and validate models.

        y: target variable to train and validate models.

        """
        print("Finding best hyperparams for Logistic Regression...")
        self.optimize_logistic_regression(X,y)
        print("Logistic Regression optimized!")

        print("\nFinding best hyperparams for Random Forest...")
        self.optimize_random_forest(X,y)
        print("Random Forest optimized!")

        print("\nFinding best hyperparams for Light GBM...")
        self.optimize_light_gbm(X,y)
        print("Light GBM optimized!")

    def optimize_logistic_regression(self,X:DataFrame,y:DataFrame) -> None:
        model_pipeline = Pipeline(steps=[
                    ('scaler', MinMaxScaler()), 
                    ('lr', LogisticRegression(class_weight='balanced', 
                                            penalty='l2',
                                            max_iter=1000,
                                            random_state=self.random_state))])
        
        param_distributions = {
            'lr__C': distributions.LogUniformDistribution(1e-4, 100),
        }
        self._best_lr, self._best_lr_params = self._optimize_model(X,y,model_pipeline, param_distributions)

    def optimize_random_forest(self,X:DataFrame,y:DataFrame) -> None:
        model_pipeline = Pipeline(steps=[
                    ('scaler', MinMaxScaler()), 
                    ('rf', RandomForestClassifier(class_weight = 'balanced', 
                                            random_state = self.random_state))])

        param_distributions = {
            'rf__n_estimators': distributions.IntUniformDistribution(10, 400),
            'rf__max_depth': distributions.IntUniformDistribution(3, 100),
            'rf__min_samples_split': distributions.IntUniformDistribution(2, 20),
            'rf__min_samples_leaf': distributions.IntUniformDistribution(2, 40),
        }
        self._best_rf, self._best_rf_params = self._optimize_model(X,y,model_pipeline, param_distributions)

    def optimize_light_gbm(self,X:DataFrame,y:DataFrame) -> None:
        model_pipeline = Pipeline(steps=[
                    ('scaler', MinMaxScaler()), 
                    ('lgbm', LGBMClassifier(class_weight='balanced', 
                                            random_state=self.random_state))])

        param_distributions = {
            'lgbm__n_estimators': distributions.IntUniformDistribution(10, 400),
            'lgbm__max_depth': distributions.IntUniformDistribution(2, 100),
            'lgbm__learning_rate': distributions.LogUniformDistribution(5e-2, 0.5),
            'lgbm__num_leaves': distributions.IntUniformDistribution(2, 50),
            'lgbm__subsample_for_bin': distributions.IntLogUniformDistribution(10, 200000),
        }
        self._best_lgbm, self._best_lgbm_params = self._optimize_model(X,y,model_pipeline, param_distributions)

    def _optimize_model(self, X, y, model_pipeline, param_distributions) -> BaseEstimator:
        optimized_model = OptunaSearchCV(model_pipeline, 
                                    param_distributions, 
                                    scoring = f1_score_micro,
                                    cv = StratifiedKFold(n_splits=self.cv_splits,shuffle=True,random_state=self.random_state), 
                                    n_trials = self.optimization_trials,
                                    verbose = 0)
                                    
        optimized_model.fit(X, y)
        return optimized_model.best_estimator_,optimized_model.best_params_

    @property
    def get_optimized_models(self):
        return self._best_lr,self._best_rf,self._best_lgbm

    @property
    def get_models_params(self):
        return self._best_lr_params, self._best_rf_params, self._best_lgbm_params

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
