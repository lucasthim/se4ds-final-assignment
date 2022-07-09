from abc import ABC, abstractmethod

from src.contracts.types import RawData


class DataLoader(ABC):
    """
    Abstract class representing data loading steps that go into production.
    """

    @abstractmethod
    def load_data(self) -> RawData:
        """Method to load the data into memory from a given path."""
        pass
