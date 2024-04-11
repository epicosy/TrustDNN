from typing import Any
from pathlib import Path
from abc import abstractmethod
from dataclasses import dataclass

SPLIT_NAMES = ['train', 'val', 'test']


@dataclass
class Split:
    name: str
    path: Path
    format: str
    headers: bool
    _features_file_name: str = 'x'
    _labels_file_name: str = 'y'
    _features: Any = None
    _labels: Any = None

    @property
    def features_file(self):
        return f"{self._features_file_name}.{self.format}"

    @property
    def labels_file(self):
        return f"{self._labels_file_name}.{self.format}"

    @property
    def features_path(self) -> Path:
        return self.path / self.features_file

    @property
    def labels_path(self) -> Path:
        return self.path / self.labels_file

    @property
    @abstractmethod
    def features(self):
        pass

    @features.setter
    def features(self, features):
        self._features = features

    @property
    @abstractmethod
    def labels(self):
        pass

    @labels.setter
    def labels(self, labels):
        self._labels = labels

    @abstractmethod
    def save(self):
        pass
