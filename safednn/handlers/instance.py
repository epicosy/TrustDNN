import time
import subprocess
import psutil
import threading

from pathlib import Path
from cement import Handler
from typing import Callable, Union
from datetime import datetime, timezone

from safednn.core.objects import Execution, Instance
from safednn.core.interfaces import HandlersInterface


def monitor_process(process, stdout_file, stderr_file, stdout_callback=None, stderr_callback=None):
    for line in process.stdout:
        with stdout_file.open('a') as f:
            f.write(line)

        if stdout_callback:
            stdout_callback(line)

    for line in process.stderr:
        with stderr_file.open('a') as f:
            f.write(line)

        if stderr_callback:
            stderr_callback(line)


def get_cpu_memory_usage(pid, cpu_usage_callback, memory_usage_callback):
    while True:
        try:
            p = psutil.Process(pid)

            with p.oneshot():
                # CPU usage
                cpu_usage_callback(p.cpu_percent())
                # Memory usage
                memory_usage_callback(p.memory_info().rss)

        except psutil.NoSuchProcess:
            break


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

        # Lists to store CPU and memory usage
        cpu_usage = []
        memory_usage = []

        # Start monitoring CPU and memory usage
        cpu_memory_thread = threading.Thread(target=get_cpu_memory_usage,
                                             args=(process.pid,
                                                   cpu_usage.append,
                                                   memory_usage.append))
        cpu_memory_thread.start()

        # Monitor stdout and stderr
        monitor_process(process, stdout_file, stderr_file,
                        stdout_callback=self.app.log.info if stdout else None,
                        stderr_callback=self.app.log.error if stderr else None)

        # Wait for the process to finish
        process.wait()

        # Wait for CPU and memory monitoring thread to finish
        cpu_memory_thread.join()

        duration = round(time.time() - start_time, 2)
        return_code = process.returncode if process.returncode is not None else -1

        if not output.exists():
            status = 'error' if return_code != 0 else 'failed'
            executed = False
        else:
            status = 'success' if return_code == 0 else 'unknown'
            executed = True

        return Execution(timestamp=timestamp, duration=duration, executed=executed, status=status, output=output,
                         return_code=return_code, mem_usage=sum(memory_usage) / (1024 ** 2),
                         cpu_usage=sum(cpu_usage) / len(cpu_usage))
