from abc import abstractmethod
from typing import Any, List, Union, Dict
from pathlib import Path

from trustdnn.handlers.plugin import PluginHandler
from trustdnn.core.dataset import Dataset
from trustdnn.core.model import Model


class BenchmarkPlugin(PluginHandler):
    class Meta:
        label = 'benchmark'

    def __init__(self, name: str, datasets_dir: str = None, models_dir: str = None, predictions_dir: str = None, **kw):
        super().__init__(name, **kw)
        self._datasets: Union[Dict[str, Dataset], None] = None
        self._models: Union[Dict[str, Model], None] = None

        self.datasets_path: Path = Path(datasets_dir).expanduser() if datasets_dir else None
        self.models_path: Path = Path(models_dir).expanduser() if models_dir else None
        self.predictions_path: Path = Path(predictions_dir).expanduser() if predictions_dir else None

        if self.datasets_path and not self.datasets_path.exists():
            raise ValueError(f"Datasets directory {datasets_dir} does not exist")

        if self.models_path and not self.models_path.exists():
            raise ValueError(f"Models directory {models_dir} does not exist")

        if self.predictions_path and not self.predictions_path.exists():
            raise ValueError(f"Predictions directory {predictions_dir} does not exist")

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

                self._datasets[f.stem] = Dataset(f)

        return self._datasets

    @property
    def models(self) -> Dict[str, Model]:
        """
            List all models
        :return:
        """
        if self._models is None:
            self._models = {}

            for f in self.models_path.iterdir():
                if f.is_dir():
                    dataset = f.name
                    for mf in f.iterdir():
                        if mf.is_file() and mf.suffix == '.h5':
                            predictions_path = self.predictions_path / dataset / f"{mf.stem}.csv"
                            self._models[mf.stem] = Model(mf, dataset, predictions_path)

        return self._models

    def get_model(self, name: str) -> Any:
        """
            Get a model by name
        :param name:
        :return:
        """
        return self.models.get(name, None)

    def get_dataset(self, name: str) -> Any:
        """
            Get a dataset by name
        :param name:
        :return:
        """
        return self.datasets.get(name, None)

    @abstractmethod
    def help(self):
        """
            Return the help message
        :return:
        """
        pass
