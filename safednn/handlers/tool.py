import os
import platform

from typing import Tuple
from pathlib import Path
from abc import abstractmethod
from safednn.handlers.plugin import PluginHandler
from safednn.core.dataset.base import Dataset
from safednn.core.model import Model


class ToolPlugin(PluginHandler):
    class Meta:
        label = 'tool'

    def __init__(self, name: str, command: str, path: str, interpreter: str = None, env_path: str = None, **kw):
        super().__init__(name, **kw)
        """
            Tool Plugin
            :param name: name of the tool
            :param command: command to run the tool
            :param path: path to the tool or working directory
            :param working_dir: working directory
            :param interpreter: interpreter to use
            :param env_path: path to the environment
        """
        # check paths
        self.command = command
        self.interpreter = interpreter
        # TODO: define default working directory
        self.path = Path(path).expanduser()

        if not self.path.exists():
            raise FileNotFoundError(f"Tool path {path} not found")

        self.env_path = Path(env_path).expanduser() if env_path else None
        self.shell = os.getenv('SHELL', '/bin/bash')
        self._activate_command = None

    @abstractmethod
    def analyze_command(self, model: Model, dataset: Dataset, working_dir: Path, **kwargs) -> Tuple[Path, str]:
        """
            Offline Analysis
        :param model: model to use
        :param dataset: dataset to use
        :param working_dir: working directory
        :param kwargs:
        :return: output path and command
        """
        pass

    @abstractmethod
    def infer_command(self, model: Model, dataset: Dataset, working_dir: Path, **kwargs) -> Tuple[Path, str]:
        """
            Inference phase to determine output
        :param model: model to use
        :param dataset: dataset to use
        :param working_dir: working directory
        :param kwargs:
        :return: output path and command
        """
        pass

    def __str__(self):
        return self.name

    @property
    def activate_command(self):
        if not self._activate_command:
            if not self.env_path:
                raise ValueError("Environment path not set")

            if not self.env_path.exists():
                raise ValueError(f"Environment {self.env_path} not found")

            if self.env_path.exists():
                if platform.system() != 'Linux':
                    raise NotImplementedError("Only Linux is supported")

                self._activate_command = f"source {self.env_path / 'bin' / 'activate'}"

        return self._activate_command

    def run_command(self, sub_command: str) -> str:
        """
            Method to build the commands for running the tool
        """

        complete_command = f"{self.command} {sub_command}"

        if self.interpreter:
            complete_command = f"{self.interpreter} {complete_command}"

        if self.env_path:
            complete_command = f"{self.activate_command} && {complete_command}"

        return f"{self.shell} -c '{complete_command}'"
