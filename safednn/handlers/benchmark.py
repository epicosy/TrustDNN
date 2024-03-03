from abc import abstractmethod
from typing import Any
from safednn.handlers.plugin import PluginHandler


class BenchmarkPlugin(PluginHandler):
    class Meta:
        label = 'benchmark'

    def __init__(self, name: str, **kw):
        super().__init__(name, **kw)
        self._datasets = []
        self._models = []

    @property
    @abstractmethod
    def datasets(self) -> list:
        """
            List all datasets
        :return:
        """
        pass

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
