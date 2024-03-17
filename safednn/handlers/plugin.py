import subprocess
import platform
import os

from cement import Handler, Interface
from pathlib import Path


class PluginsInterface(Interface):
    """
        Handlers' Interface
    """
    class Meta:
        """
            Meta class
        """
        interface = 'plugins'


class PluginHandler(PluginsInterface, Handler):
    class Meta:
        label = 'plugin'

    def __init__(self, name: str = None, env_path: str = None, **kw):
        super().__init__(**kw)
        self.name = name
        self.env_path = Path(env_path) if env_path else None
        self.shell = os.getenv('SHELL', '/bin/bash')

    def __str__(self):
        return self.name

    def activate_environment(self):
        """
        Activate the virtual environment for the plugin
        """
        if self.env_path and self.env_path.exists():
            activate_script = "activate" if platform.system() == "Windows" else "bin/activate"
            activate_path = os.path.join(self.env_path, "bin" if 'bash' in self.shell else "Scripts", activate_script)
            activate_command = f"{self.shell} -c 'source {activate_path}'" if 'bash' in self.shell else f"{activate_path}"
            subprocess.run(activate_command, shell=True)
        else:
            self.app.log.error(f"Environment {self.env_path} not found")

    def _run_command(self, command: str, path: Path):
        """
        Helper function to run TrustBench commands
        """

        activate_path = os.path.join(self.env_path, "bin/activate")
        activate_command = f"{self.shell} -c 'source {activate_path}'" if 'bash' in self.shell else f"{activate_path}"
        #subprocess.run(activate_command, shell=True)

        trustbench_command = f"{self.env_path}/bin/python -m {command}"
        result = subprocess.run(f"{activate_command} && {trustbench_command}",
                                shell=True, cwd=str(path.expanduser()), capture_output=True, text=True)

        # Construct the path to the Python interpreter inside the virtual environment
        #python_interpreter = os.path.join(self.env_path, "bin", "python")

        # Run TrustBench command using the virtual environment's Python interpreter
        #result = subprocess.run([python_interpreter, "-m", "trustbench"] + command.split(),
        #                        cwd=str(path.expanduser()), capture_output=True, text=True)

        if result.stdout:
            self.app.log.info(result.stdout)
        if result.stderr:
            self.app.log.error(result.stderr)
