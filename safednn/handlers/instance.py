import time
import subprocess

from pathlib import Path
from cement import Handler
from typing import Callable, Union
from datetime import datetime, timezone

from safednn.core.objects import Execution, Instance
from safednn.core.interfaces import HandlersInterface


class InstanceHandler(HandlersInterface, Handler):
    class Meta:
        label = 'instance'

    def __init__(self, **kw):
        super().__init__(**kw)

    def __call__(self, instance: Instance, command_call: Callable, sub_command_call: Callable,
                 tool_path: Path) -> Union[Execution, None]:
        out_path, sub_command = sub_command_call(instance.model, instance.dataset, instance.working_dir)
        command = command_call(sub_command)

        if out_path and not out_path.exists():
            return self._execute(command, instance.working_dir.parent, output=out_path, cwd=tool_path, stdout=True,
                                 stderr=True)

        return None

    def _execute(self, command: str, log_path: Path, output: Path, cwd: Path, stdout: bool = False,
                 stderr: bool = False) -> Execution:
        """
            Function to run shell commands
        """

        self.app.log.info(f"Executing: {command}")
        timestamp = int(datetime.now(timezone.utc).timestamp())
        start_time = time.time()

        stdout_file = log_path / f"{timestamp}.stdout"
        stderr_file = log_path / f"{timestamp}.stderr"

        # Execute the command with stdout redirected to a pipe
        process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                   text=True, cwd=cwd)

        # Print stdout and stderr lines as they become available
        for line in process.stdout:
            with stdout_file.open('a') as f:
                f.write(line)

            if stdout:
                self.app.log.info(line.rstrip())

        process.wait()

        for line in process.stderr:
            with stderr_file.open('a') as f:
                f.write(line)

            if stderr:
                self.app.log.error(line.rstrip())

        duration = round(time.time() - start_time, 2)
        return_code = process.returncode if process.returncode is not None else -1

        if not output.exists():
            status = 'error' if return_code != 0 else 'failed'
            executed = False
        else:
            status = 'success' if return_code == 0 else 'unknown'
            executed = True

        return Execution(timestamp=timestamp, duration=duration, executed=executed, status=status, output=output,
                         return_code=return_code)
