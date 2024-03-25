import pandas as pd

from pathlib import Path
from cement import Controller, ex
from safednn.core.evaluation import Evaluation
from safednn.handlers.benchmark import BenchmarkPlugin
from safednn.handlers.tool import ToolPlugin
from safednn.core.exc import SafeDNNError
from safednn.core.plotter import Plotter


class Evaluate(Controller):
    class Meta:
        label = 'evaluate'
        stacked_on = 'base'
        stacked_type = 'nested'

        # text displayed at the top of --help output
        description = 'Command for executing a tool on a benchmark.'

        # text displayed at the bottom of --help output
        epilog = 'Usage: safednn evaluate -wd ~/workdir efficiency'

        # controller level arguments. ex: 'safednn --version'
        arguments = [
            (['-wd', '--workdir'], {'help': 'Working directory', 'type': str, 'required': True})
        ]

    def __init__(self, **kw):
        super().__init__(**kw)
        self._working_dir = None
        self._benchmarks = None
        self._tools = None

    @property
    def tools(self):
        if self._tools is None:
            self._tools = {}

        return self._tools

    @property
    def benchmarks(self):
        if self._benchmarks is None:
            self._benchmarks = {}

        return self._benchmarks

    def get_benchmark(self, name: str):
        if name not in self.benchmarks:
            self.benchmarks[name] = self.app.get_plugin_handler(name=name, kind=BenchmarkPlugin)

        return self.benchmarks[name]

    def get_tool(self, name: str):
        if name not in self.tools:
            self.tools[name] = self.app.get_plugin_handler(name=name, kind=ToolPlugin)

        return self.tools[name]

    def get_notifications(self, tool_name: str, output_path: Path):
        tool = self.get_tool(tool_name)

        return tool.get_notifications(output_path)

    def get_test_labels(self, dataset_name: str, benchmark_name: str):
        benchmark = self.get_benchmark(benchmark_name)
        dataset = benchmark.get_dataset(dataset_name)

        if dataset is None:
            raise SafeDNNError(f"Dataset {dataset_name} not found in benchmark {benchmark_name}")

        return dataset.test.labels

    def get_predictions(self, model_name: str, benchmark: str):
        benchmark = self.get_benchmark(benchmark)
        model = benchmark.get_model(model_name)

        if model is None:
            raise SafeDNNError(f"Model {model_name} not found in benchmark {benchmark}")

        return model.predictions

    def _parse_working_dir(self):
        self._working_dir = Path(self.app.pargs.workdir).expanduser()

    def _post_argument_parsing(self):
        if self.app.pargs.__controller_namespace__ == self.Meta.label:
            self._parse_working_dir()

    @property
    def working_dir(self):
        return self._working_dir

    def _default(self):
        """Default action if no sub-command is passed."""

        self.app.args.print_help()

    @ex(
        help='Computes the efficiency of the executions under a working directory',
        arguments=[
            (['-f', '--force'], {'help': 'Force re-computation of results', 'action': 'store_true'})
        ]
    )
    def efficiency(self):
        efficiency_path = self.working_dir / "efficiency.csv"

        if efficiency_path.exists() and not self.app.pargs.force:
            self.app.log.error(f"Efficiency file already exists in {efficiency_path}")
            exit(0)

        executions_path = self.working_dir / "executions.csv"

        if not executions_path.exists():
            self.app.log.error(f"Executions file not found in {executions_path}")
            exit(1)

        executions = pd.read_csv(executions_path, index_col=False)
        results = []

        for tool_phase, rows in executions.groupby(['tool', 'phase']):
            tool, phase = tool_phase
            average_duration = rows['duration'].mean()
            results.append({
                'tool': tool,
                'phase': phase,
                'duration': average_duration
            })

        pd.DataFrame(results).to_csv(self.working_dir / "efficiency.csv", index=False)

    @ex(
        help='Computes the effectiveness of the executions under a working directory',
        arguments=[
            (['-f', '--force'], {'help': 'Force re-computation of results', 'action': 'store_true'}),
            (['-i', '--invert'], {'help': 'Sets the positive class the mis-classifications', 'action': 'store_true'})
        ]
    )
    def effectiveness(self):
        effectiveness_path = self.working_dir / "effectiveness.csv"

        if effectiveness_path.exists() and not self.app.pargs.force:
            self.app.log.error(f"Effectiveness file already exists in {effectiveness_path}")
            exit(0)

        executions_path = self.working_dir / "executions.csv"

        if not executions_path.exists():
            self.app.log.error(f"Executions file not found in {executions_path}")
            exit(1)

        executions = pd.read_csv(executions_path, index_col=False)
        infer_success_executions = executions[(executions['phase'] == 'infer') & (executions['status'] == 'success')]
        results = []

        for group, rows in infer_success_executions.groupby(['tool', 'model', 'dataset', 'benchmark']):
            tool, model, dataset, benchmark = group
            labels = self.get_test_labels(dataset, benchmark)
            predictions = self.get_predictions(model, benchmark)

            tool_model_run_effectiveness = []

            for i, row in rows.iterrows():
                notifications = self.get_notifications(tool, row['output'])
                evaluation = Evaluation(notifications=notifications, labels=labels, predictions=predictions,
                                        invert=self.app.pargs.invert)

                effectiveness = evaluation.performance()
                effectiveness.update(evaluation.to_dict())
                tool_model_run_effectiveness.append(effectiveness)

            df = pd.DataFrame(tool_model_run_effectiveness)
            # Compute the average effectiveness for the tool and model
            tool_model_effectiveness = df.mean().to_dict()
            tool_model_effectiveness['tool'] = tool
            tool_model_effectiveness['model'] = model
            results.append(tool_model_effectiveness)

        pd.DataFrame(results).to_csv(self.working_dir / "effectiveness.csv", index=False)

    @ex(
        help='Computes the variance of the executions under a working directory',
        arguments=[
            (['-f', '--force'], {'help': 'Force re-computation of results', 'action': 'store_true'}),
            (['-i', '--invert'], {'help': 'Sets the positive class the mis-classifications', 'action': 'store_true'})
        ]
    )
    def variance(self):
        variance_path = self.working_dir / "variance.csv"

        if variance_path.exists() and not self.app.pargs.force:
            self.app.log.error(f"Variance file already exists in {variance_path}")
            exit(0)

        executions_path = self.working_dir / "executions.csv"

        if not executions_path.exists():
            self.app.log.error(f"Executions file not found in {executions_path}")
            exit(1)

        plotter = Plotter(figures_path=self.working_dir)

        executions = pd.read_csv(executions_path, index_col=False)
        infer_success_executions = executions[(executions['phase'] == 'infer') & (executions['status'] == 'success')]
        results = []

        for tool_model, rows in infer_success_executions.groupby(['tool', 'model']):
            tool, model = tool_model

            for i, row in rows.iterrows():
                notifications = self.get_notifications(tool, row['output'])
                labels = self.get_test_labels(row['dataset'], row['benchmark'])
                predictions = self.get_predictions(model, row['benchmark'])
                evaluation = Evaluation(notifications=notifications, labels=labels, predictions=predictions,
                                        invert=self.app.pargs.invert)

                effectiveness = evaluation.performance()
                effectiveness.update(evaluation.to_dict())
                effectiveness['tool'] = tool
                effectiveness['model'] = model
                effectiveness['run'] = i

                results.append(effectiveness)

        df = pd.DataFrame(results)
        plotter.box_plot(df, x='tool', y='mcc', tag='variance', hue='model')

        #.to_csv(self.working_dir / "effectiveness.csv", index=False))
