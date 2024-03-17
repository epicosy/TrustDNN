import numpy as np
import pandas as pd

from pathlib import Path
from dataclasses import dataclass


# TODO: make an split interface and implement csv and npy splits
@dataclass
class Split:
    name: str
    _features_file: str
    _labels_file: str
    headers: bool = True
    _features: pd.DataFrame = None
    _labels: np.ndarray = None
    _path: Path = None
    _format: str = 'csv'

    @property
    def path(self):
        return self._path

    @path.setter
    def path(self, path: Path):
        self._path = path / self.name

    @property
    def features(self):
        if self._features is None:
            if self._format == 'npy':
                self._features = np.load(str(self.path / self._features_file))
            else:
                self._features = pd.read_csv(str(self.path / self._features_file), delimiter=',', encoding='utf-8',
                                             header=None if not self.headers else 'infer')

        return self._features

    @features.setter
    def features(self, features):
        self._features = features

    @property
    def labels(self):
        if self._labels is None:
            if self._format == 'npy':
                self._labels = np.load(str(self.path / self._labels_file))
            else:
                self._labels = np.loadtxt(str(self.path / self._labels_file), dtype=int)

        return self._labels

    @labels.setter
    def labels(self, labels):
        self._labels = labels

    def save(self):
        self.path.mkdir(parents=True, exist_ok=True)
        if self._format == 'csv':
            self._features.to_csv(str(self.path / self._features_file), index=False, header=self.headers)
            np.savetxt(str(self.path / self._labels_file), self._labels, fmt='%d')
        else:
            np.save(str(self.path / self._features_file), self._features)
            np.save(str(self.path / self._labels_file), self._labels)


# TODO: move the splits below to a separate file/module
@dataclass
class Train(Split):
    name: str = 'train'
    _features_file: str = 'x.csv'
    _labels_file: str = 'y.csv'
    headers: bool = False


@dataclass
class Val(Split):
    name: str = 'val'
    _features_file: str = 'x.csv'
    _labels_file: str = 'y.csv'
    headers: bool = True


@dataclass
class Test(Split):
    name: str = 'test'
    _features_file: str = 'x.csv'
    _labels_file: str = 'y.csv'
    headers: bool = True

