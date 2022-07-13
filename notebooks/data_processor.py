import pandas as pd
from sklearn.model_selection import train_test_split

class DataProcessor():
    
    def __init__(self,features,target) -> None:
        pass

    def preprocess(self,df) -> pd.DataFrame:
        pass

    def train_test_split(self,df:pd.DataFrame,train_size = 1500,random_state = 42) -> pd.DataFrame:    
        df_train,df_test = train_test_split(df,train_size=train_size,random_state=random_state)
        return df_train,df_test
    
    def _encode_features(self,df) -> pd.DataFrame:
        pass