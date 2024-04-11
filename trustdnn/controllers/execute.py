import pandas as pd

from typing import Dict
from pathlib import Path
from cement import Controller, ex

from trustdnn.handlers.benchmark import BenchmarkPlugin
from trustdnn.handlers.tool import ToolPlugin
from trustdnn.core.objects import Execution, Instance
from trustdnn.core.dataset import Dataset
from trustdnn.core.model import Model


class Execute(Controller):
    class Meta:
        label = 'execute'
        stacked_on = 'base'
        stacked_type = 'nested'

        # text displayed at the top of --help output
        description = 'Command for executing a tool on a benchmark.'

        # text displayed at the bottom of --help output
        epilog = 'Usage: trustdnn execute --tool prophecy --benchmark mnist --dataset mnist --model model1 analyze'

        # controller level arguments. ex: 'trustdnn --version'
        arguments = [
            (['-wd', '--workdir'], {'help': 'Working directory', 'type': str, 'required': False}),
            (['-b', '--benchmark'], {'help': 'Benchmark name', 'type': str, 'required': True}),
            (['-t', '--tool'], {'help': 'Tool name', 'type': str, 'required': True}),
            (['-id'], {'help': 'Identifier for execution.',  'type': str, 'required': True}),
            (['-d', '--datasets'], {'help': 'Dataset name', 'nargs': "*", 'type': str, 'required': False}),
            (['-m', '--models'], {'help': 'Target model', 'nargs': "*", 'type': str, 'required': False})
        ]

    def __init__(self, **kw):
        super().__init__(**kw)
        self._benchmark = None
        self._datasets = None
        self._instances = None
        self._working_dir = None
        self._tool_working_dir = None

    @property
    def instances(self) -> list:
        return self._instances

    def _parse_datasets(self) -> Dict[str, Dataset]:
        datasets = {}

        if self.app.pargs.datasets:
            for dataset_name in self.app.pargs.datasets:
                if dataset_name in self.benchmark.datasets:
                    datasets[dataset_name] = self.benchmark.datasets[dataset_name]
                else:
                    self.app.log.error(f"Dataset {dataset_name} not found in benchmark {self.app.pargs.benchmark}")
        else:
            datasets = self.benchmark.datasets

        return datasets

    def _parse_models(self) -> Dict[str, Model]:
        models = {}

        if self.app.pargs.models:
            for model_name in self.app.pargs.models:
                if model_name in self.benchmark.models:
                    models[model_name] = self.benchmark.models[model_name]
                else:
                    self.app.log.error(f"Model {model_name} not found in benchmark {self.app.pargs.benchmark}")
        else:
            models = self.benchmark.models

        return models

    def _init_instances(self):
        datasets = self._parse_datasets()
        models = self._parse_models()

        self._instances = []

        for dataset in datasets.values():
            for model in models.values():
                if dataset.name == model.dataset:
                    instance = Instance(dataset=dataset, model=model, working_dir=self._tool_working_dir / model.name,
                                        phase=self.app.pargs.command)
                    instance.working_dir.mkdir(parents=True, exist_ok=True)

                    self._instances.append(instance)

    def _parse_working_dirs(self):
        if self.app.pargs.workdir:
            self._working_dir = Path(self.app.pargs.workdir).expanduser()
        else:
            self._working_dir = Path.cwd() / 'workdir'

        self._working_dir.mkdir(parents=True, exist_ok=True)
        self._tool_working_dir = self._working_dir / self.app.pargs.tool

        if self.app.pargs.id:
            self._tool_working_dir = self._tool_working_dir / self.app.pargs.id

        self._tool_working_dir.mkdir(parents=True, exist_ok=True)

    def _post_argument_parsing(self):
        if self.app.pargs.__controller_namespace__ == self.Meta.label:
            self._parse_working_dirs()

            self._tool = self.app.get_plugin_handler(name=self.app.pargs.tool, kind=ToolPlugin)
            self._benchmark = self.app.get_plugin_handler(name=self.app.pargs.benchmark, kind=BenchmarkPlugin)
            self._init_instances()

    @property
    def working_dir(self):
        return self._working_dir

    @property
    def tool_working_dir(self):
        return self._tool_working_dir

    @property
    def benchmark(self):
        return self._benchmark

    def save_execution(self, instance: Instance, execution: Execution):
        if execution.status == 'exists':
            return

        path = self.working_dir / "executions.csv"

        execution = execution.to_dict()
        execution['tool'] = self.app.pargs.tool
        execution['benchmark'] = self.app.pargs.benchmark
        execution['dataset'] = instance.dataset.name
        execution['model'] = instance.model.name
        execution['phase'] = instance.phase
        execution['output'] = str(execution['output'])

        if path.exists():
            executions = pd.read_csv(path, index_col=False)
            executions = executions.append(execution, ignore_index=True)
        else:
            executions = pd.DataFrame([execution])

        executions.to_csv(path, index=False)

    def _default(self):
        """Default action if no sub-command is passed."""

        self.app.args.print_help()

    @ex(
        help='Offline analysis of a tool on a dataset from a given benchmark'
    )
    def analyze(self):
        instance_handler = self.app.handler.get('handlers', 'instance', setup=True)

        for instance in self.instances:
            self.app.log.info(f"Running inference on {instance}")
            execution = instance_handler(instance=instance, command_call=self._tool.run_command,
                                         tool_path=self._tool.path, sub_command_call=self._tool.analyze_command)
            if execution:
                self.save_execution(instance, execution)

    @ex(
        help='Runs a tool on a dataset from a given benchmark'
    )
    def infer(self):
        instance_handler = self.app.handler.get('handlers', 'instance', setup=True)

        for instance in self.instances:
            self.app.log.info(f"Running inference on {instance}")

            execution = instance_handler(instance=instance, command_call=self._tool.run_command,
                                         tool_path=self._tool.path, sub_command_call=self._tool.infer_command)

            if execution:
                self.save_execution(instance, execution)
