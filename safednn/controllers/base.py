from cement import Controller, ex
from cement.utils.version import get_version_banner
from ..core.version import get_version

from safednn.handlers.benchmark import BenchmarkPlugin
from safednn.handlers.tool import ToolPlugin


VERSION_BANNER = """
A framework for evaluating tools that reason about the trustworthiness of the DNN's predictions. %s
%s
""" % (get_version(), get_version_banner())


class Base(Controller):
    class Meta:
        label = 'base'

        # text displayed at the top of --help output
        description = 'A framework for evaluating tools that reason about the trustworthiness of DNNs.'

        # text displayed at the bottom of --help output
        epilog = 'Usage: safednn command1 --foo bar'

        # controller level arguments. ex: 'safednn --version'
        arguments = [
            (['-v', '--version'], {'action': 'version', 'version': VERSION_BANNER}),
        ]

    def _default(self):
        """Default action if no sub-command is passed."""

        self.app.args.print_help()

    @ex(
        help='Runs a tool on a dataset from a given benchmark',

        arguments=[
            (['-b', '--benchmark'], {'help': 'Benchmark name', 'action': 'store', 'required': True}),
            (['-d', '--dataset'], {'help': 'Dataset name', 'action': 'store', 'required': True}),
            (['-t', '--tool'], {'help': 'Tool name', 'action': 'store', 'required': True}),
            (['-m', '--model'], {'help': 'Target model', 'action': 'store', 'required': True})
        ]
    )
    def run(self):
        """Example sub-command."""

        self.app.log.info(f'Running tool {self.app.pargs.tool} on dataset {self.app.pargs.dataset} from benchmark '
                          f'{self.app.pargs.benchmark}')

        benchmark = self.app.get_plugin_handler(self.app.pargs.benchmark, BenchmarkPlugin)
        tool = self.app.get_plugin_handler(self.app.pargs.tool, ToolPlugin)
        print(benchmark.datasets)
