import os
import platform
import subprocess

from pathlib import Path
from abc import abstractmethod
from subprocess import Popen
from safednn.handlers.plugin import PluginHandler
from safednn.core.dataset.base import Dataset


class ToolPlugin(PluginHandler):
    class Meta:
        label = 'tool'

    def __init__(self, name: str, command: str, path: str, env_path: str = None, **kw):
        super().__init__(name, **kw)
        """
            Tool Plugin
            :param name: name of the tool
            :param command: command to run the tool
            :param path: path to the tool or working directory
            :param env_path: path to the environment
        """
        # check paths
        self.command = command
        # TODO: define default working directory
        self.path = Path(path).expanduser()

        if not self.path.exists():
            raise FileNotFoundError(f"Tool path {path} does not exist")

        self.env_path = Path(env_path).expanduser() if env_path else None
        self.shell = os.getenv('SHELL', '/bin/bash')
        self._activate_command = None

    @abstractmethod
    def get_results(self, **kwargs):
        """
            Get the results
        :param kwargs:
        :return:
        """
        pass

    @abstractmethod
    def run(self, model: str, dataset: Dataset, **kwargs):
        """
            Run the tool
        :param model: model to use
        :param dataset: dataset to use
        :param kwargs:
        :return:
        """
        pass

    def __str__(self):
        return self.help()

    @abstractmethod
    def help(self):
        """
            Return the help message
        :return:
        """
        pass

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

    def run_command(self, sub_command: str, interpreter: str = None, use_env: bool = True, stdout: bool = False,
                    stderr: bool = False) -> Popen:
        """
        Helper function to run shell commands
        """

        complete_command = f"{self.command} {sub_command}"

        if interpreter:
            complete_command = f"{interpreter} {complete_command}"

        if use_env:
            complete_command = f"{self.activate_command} && {complete_command}"

        complete_command = f"{self.shell} -c '{complete_command}'"
        self.app.log.info(f"Executing: {complete_command}")
        # Execute the command with stdout redirected to a pipe
        process = subprocess.Popen(complete_command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                   text=True, cwd=str(self.path))

        # Print stdout and stderr lines as they become available
        for line in process.stdout:
            if stdout:
                self.app.log.info(line.rstrip())

        for line in process.stderr:
            if stderr:
                self.app.log.error(line.rstrip())

        return process
