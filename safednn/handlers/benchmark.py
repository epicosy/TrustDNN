from abc import abstractmethod
from typing import Any, List, Union, Dict
from pathlib import Path

from safednn.handlers.plugin import PluginHandler
from safednn.core.dataset import Dataset


class BenchmarkPlugin(PluginHandler):
    class Meta:
        label = 'benchmark'

    def __init__(self, name: str, datasets_dir: str, **kw):
        super().__init__(name, **kw)
        self._datasets: Union[Dict[str, Dataset], None] = None
        self._models = []

        if datasets_dir is None:
            raise ValueError("Datasets directory is required")

        self.datasets_path: Path = Path(datasets_dir).expanduser()

        if not self.datasets_path.exists():
            raise ValueError(f"Datasets directory {datasets_dir} does not exist")

    @property
    def datasets(self) -> Dict[str, Dataset]:
        """
            List all datasets
        :return:
        """
        if self._datasets is None:
            self._datasets = {}

            for f in self.datasets_path.iterdir():
                if not f.is_dir():
                    continue

                self._datasets[f.name] = Dataset(f)

        return self._datasets

    @property
    @abstractmethod
    def models(self) -> list:
        """
            List all models
        :return:
        """
        pass

    @abstractmethod
    def get_model(self, name: str) -> Any:
        """
            Get a model by name
        :param name:
        :return:
        """
        pass

    @abstractmethod
    def get_dataset(self, name: str) -> Any:
        """
            Get a dataset by name
        :param name:
        :return:
        """
        pass

    @abstractmethod
    def help(self):
        """
            Return the help message
        :return:
        """
        pass
