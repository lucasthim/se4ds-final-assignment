import pandas as pd

from src.contracts.data_loader import DataLoader
from src.contracts.data_processor import DataProcessor
from src.contracts.types import RawData


class CompoundDataLoader(DataLoader):
    def __init__(self, path_stores_data, path_sales_data) -> None:
        self.path_stores_data = path_stores_data
        self.path_sales_data = path_sales_data

    def load_data(self) -> RawData:
        stores_data = self._load_stores_data(self.path_stores_data)
        sales_data = self._load_sales_data(self.path_sales_data)
        return stores_data, sales_data

    def _load_stores_data(self, path_stores_data):
        return pd.read_csv(path_stores_data)

    def _load_sales_data(self, path_sales_data):
        return pd.read_csv(path_sales_data)


class CompoundDataProcessor(DataProcessor):
    def __init__(self, max_sales_value: float) -> None:
        self.max_sales_value = max_sales_value

    def preprocess(self, raw_data: tuple) -> RawData:
        stores_data, sales_data = raw_data
        joint_data = self._join_dataframes(stores_data, sales_data)
        encoded_stores_data = self._encode_stores(joint_data)
        non_negative_sales_data = self._remove_negative_sales(encoded_stores_data)
        processed_data = self._clip_max_sales_value(non_negative_sales_data)

        return processed_data

    def _join_dataframes(self, stores_data, sales_data):
        joint_data = pd.merge(stores_data, sales_data, on="store_id", how="right")
        return joint_data

    def _encode_stores(self, data: pd.DataFrame) -> pd.DataFrame:
        encoded_data = data.copy()  # simulate encoding step or anything else.
        return encoded_data

    def _remove_negative_sales(self, data: pd.DataFrame) -> pd.DataFrame:
        non_negative_sales_data = data.copy()
        non_negative_sales_data.loc[non_negative_sales_data["sales"] < 0, "sales"] = 0
        return non_negative_sales_data

    def _clip_max_sales_value(self, data: pd.DataFrame) -> pd.DataFrame:
        clipped_data = data.copy()
        clipped_data = data[data["sales"] > self.max_sales_value] = self.max_sales_value
        return clipped_data


if __name__ == "__main__":

    compound_data_loader = CompoundDataLoader(path_stores_data="./data/stores.csv", path_sales_data="./data/sales.csv")
    raw_data = compound_data_loader.load_data()

    compound_data_processor = CompoundDataProcessor(max_sales_value=100)
    processed_data = compound_data_processor.preprocess(raw_data=raw_data)
