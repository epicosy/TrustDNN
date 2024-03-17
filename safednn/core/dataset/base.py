from abc import abstractmethod
from pathlib import Path
from typing import Dict

from safednn.core.split.base import Split


class Dataset:
    def __init__(self, path: Path, splits: Dict[str, Split], data_format: str = 'csv'):
        """
        :param path: Path to the raw dataset
        :param data_format: File format of the dataset
        """
        self.name = path.name
        self.format = data_format
        self.root_path = path
        self._data = None
        # TODO: splits should be loaded instead of passed as a parameter
        self.splits = splits

    @property
    def data(self):
        return self._data

    @data.setter
    @abstractmethod
    def data(self, data):
        pass
