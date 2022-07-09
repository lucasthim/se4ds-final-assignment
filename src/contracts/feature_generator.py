from abc import ABC, abstractmethod

from src.contracts.types import FeatureData, ProcessedData, TargetData


class FeatureGenerator(ABC):
    """
    Abstract class representing feature generation process that goes into production.

    PS: A process that needs to keep a state (such as a sklearn Transformer) in order
        to transform data should NOT be kept here. It should be kept in the ModelPipeline class.
    """

    @abstractmethod
    def get_features(self, data: ProcessedData) -> FeatureData:
        """Method to generate feature data."""
        pass

    @abstractmethod
    def get_target(self, data: ProcessedData) -> TargetData:
        """Method to generate target data."""
        pass
