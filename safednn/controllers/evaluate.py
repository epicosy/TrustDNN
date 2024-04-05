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
        self._plotter = None

    @property
    def plotter(self):
        return self._plotter

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
            self._plotter = Plotter(figures_path=self.working_dir)


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

        for tool_phase_dataset, rows in executions.groupby(['tool', 'phase', 'dataset']):
            tool, phase, dataset = tool_phase_dataset
            average_duration = rows['duration'].mean()
            results.append({
                'tool': tool,
                'phase': phase,
                'dataset': dataset,
                'duration': average_duration
            })

        df = pd.DataFrame(results)
        df.to_csv(self.working_dir / "efficiency.csv", index=False)
        self.plotter.fig_size = (11, 9)
        self.plotter.stacked_bar_plot(df, x='dataset', y='duration', stack='phase', hue='tool', y_label='Duration (s)',
                                      tag='efficiency', x_label='Tool')

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

        for tool_model, rows in infer_success_executions.groupby(['tool', 'model']):
            tool_name, model = tool_model
            tool = self.get_tool(tool_name)

            for i, row in rows.iterrows():
                if not tool.has_metrics:
                    notifications = self.get_notifications(tool_name, row['output'])
                    labels = self.get_test_labels(row['dataset'], row['benchmark'])
                    predictions = self.get_predictions(model, row['benchmark'])
                    evaluation = Evaluation(notifications=notifications, labels=labels, predictions=predictions,
                                            invert=self.app.pargs.invert)

                    effectiveness = evaluation.performance()
                    effectiveness.update(evaluation.to_dict())
                else:
                    effectiveness = tool.get_metrics(row['output'])
                    effectiveness['correct'] = None
                    effectiveness['incorrect'] = None
                    effectiveness['uncertain'] = None

                effectiveness['tool'] = tool_name
                effectiveness['model'] = model
                effectiveness['run'] = i

                results.append(effectiveness)

        df = pd.DataFrame(results)
        df.to_csv(self.working_dir / "effectiveness.csv", index=False)
        self.plotter.fig_size = (20, 10)
        self.plotter.bar_plot(df, x='model', y='mcc', hue='tool', y_label='MCC', tag='effectiveness', x_label='Models',
                              error_bars=True)
