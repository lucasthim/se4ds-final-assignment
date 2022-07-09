from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, List


@dataclass
class Artifact:
    name: str
    object: Any  # TODO: modify name
    params: dict
    # path: str = ""


class ArtifactsHandler(ABC):
    @abstractmethod
    def save(self, artifacts: List[Artifact]):
        pass

    @abstractmethod
    def load(self) -> List[Artifact]:
        pass
