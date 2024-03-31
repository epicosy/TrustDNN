from cement import Controller
from cement.utils.version import get_version_banner
from ..core.version import get_version


VERSION_BANNER = "A framework for evaluating tools that reason about the trustworthiness of the DNN's predictions."


class Base(Controller):
    class Meta:
        label = 'base'

        # text displayed at the top of --help output
        description = 'A framework for evaluating tools that reason about the trustworthiness of DNNs.'

        # text displayed at the bottom of --help output
        epilog = 'Usage: safednn command --option 1'

        # controller level arguments. ex: 'safednn --version'
        arguments = [
            (['-v', '--version'], {'action': 'version', 'version': VERSION_BANNER}),
        ]

    def _default(self):
        """Default action if no sub-command is passed."""

        self.app.args.print_help()
