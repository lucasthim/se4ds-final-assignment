from abc import ABC, abstractmethod

from src.contracts.types import ProcessedData, RawData


class DataProcessor(ABC):
    """
    Abstract class representing data processing steps that go into production.
    """

    @abstractmethod
    def preprocess(self, raw_data: RawData) -> ProcessedData:
        """Method to preprocess the data."""
        pass
