import pandas as pd

from pathlib import Path
from cement import Controller, ex
from trustdnn.core.evaluation import Evaluation
from trustdnn.handlers.benchmark import BenchmarkPlugin
from trustdnn.handlers.tool import ToolPlugin
from trustdnn.core.exc import TrustDNNError
from trustdnn.core.plotter import Plotter


class Evaluate(Controller):
    class Meta:
        label = 'evaluate'
        stacked_on = 'base'
        stacked_type = 'nested'

        # text displayed at the top of --help output
        description = 'Command for executing a tool on a benchmark.'

        # text displayed at the bottom of --help output
        epilog = 'Usage: trustdnn evaluate -wd ~/workdir efficiency'

        # controller level arguments. ex: 'trustdnn --version'
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
            raise TrustDNNError(f"Dataset {dataset_name} not found in benchmark {benchmark_name}")

        return dataset.test.labels

    def get_predictions(self, model_name: str, benchmark: str):
        benchmark = self.get_benchmark(benchmark)
        model = benchmark.get_model(model_name)

        if model is None:
            raise TrustDNNError(f"Model {model_name} not found in benchmark {benchmark}")

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
        successful_executions = executions[executions['status'] == 'success']

        results = []

        for tool_phase_dataset, rows in successful_executions.groupby(['tool', 'phase', 'dataset']):
            tool, phase, dataset = tool_phase_dataset
            average_duration = round(rows['duration'].mean(), 2)
            average_memory = round(rows['mem_peak'].mean() / (1024**2), 2)
            results.append({
                'tool': tool,
                'phase': phase,
                'dataset': dataset,
                'duration': average_duration,
                'memory': average_memory
            })

        df = pd.DataFrame(results)
        df.to_csv(self.working_dir / "efficiency.csv", index=False)
        self.plotter.fig_size = (11, 9)
        self.plotter.stacked_bar_plot(df, x='dataset', y='duration', stack='phase', hue='tool', y_label='Duration (s)',
                                      tag='efficiency', x_label='Tool')
        self.plotter.stacked_bar_plot(df, x='dataset', y='memory', stack='phase', hue='tool', y_label='Memory (MiB)',
                                      tag='efficiency_memory', x_label='Tool')

    @ex(
        help='Computes the effectiveness of the executions under a working directory',
        arguments=[
            (['-f', '--force'], {'help': 'Force re-computation of results', 'action': 'store_true'}),
            (['-i', '--invert'], {'help': 'Sets the positive class the mis-classifications', 'action': 'store_true'}),
            (['-rwd', '--replace_workdir'], {'help': 'Replace the working dir (old:new)', 'type': str})
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

        print(f"Parsing {len(infer_success_executions)} successful executions")

        if self.app.pargs.replace_workdir:
            old, new = self.app.pargs.replace_workdir.split(':')
            infer_success_executions['output'] = infer_success_executions['output'].str.replace(old, new)
            # executions.to_csv(executions_path, index=False)

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

        # select the best for each tool and model by mcc score
        best = df.sort_values('mcc', ascending=False).groupby(['tool', 'model']).first().reset_index()
        best.to_csv(self.working_dir / "best.csv", index=False)

        df.to_csv(self.working_dir / "effectiveness.csv", index=False)
        self.plotter.fig_size = (27, 7)
        self.plotter.bar_plot(df, x='model', y='mcc', hue='tool', y_label='MCC', tag='effectiveness', x_label='Models',
                              error_bars=True)
