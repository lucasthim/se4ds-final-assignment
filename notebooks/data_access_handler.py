from unicodedata import name
import pandas as pd

class DataAccessHandler():

    """ Class that provides access to the dataset."""
    
    def __init__(self,main_path:str) -> None:
        """
        Params:

        main_path: local path to load data or save data.
        
        """

        self.main_path = main_path if (main_path[-1] == "/") else main_path + "/"

    def load(self, dataset_type:str) -> pd.DataFrame:
        """Load dataset into memory. dataset_type parameter is to filter specific sets, such as train and test."""
        dataset_name = f"fetal_health.csv" if dataset_type == "" else f"fetal_health_{dataset_type}.csv"
        return pd.read_csv(self.main_path + dataset_name)

    def save(self, df:pd.DataFrame, dataset_type:str) -> None:
        """Save dataset into main_path directory. dataset_type parameter is to save specific sets, such as train and test."""
        dataset_name = f"fetal_health.csv" if dataset_type == "" else f"fetal_health_{dataset_type}.csv"
        df.to_csv(self.main_path + dataset_name,index=False)

