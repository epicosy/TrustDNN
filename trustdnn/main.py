
from pathlib import Path
from cement import App, TestApp, init_defaults
from cement.core.exc import CaughtSignal, InterfaceError
from .core.exc import TrustDNNError
from .controllers.base import Base
from .controllers.execute import Execute
from .controllers.evaluate import Evaluate

from trustdnn.core.interfaces import PluginsInterface, HandlersInterface
from trustdnn.handlers.instance import InstanceHandler

from trustdnn.handlers.tool import ToolPlugin
from trustdnn.handlers.benchmark import BenchmarkPlugin


# configuration defaults
CONFIG = init_defaults('trustdnn')


class TrustDNN(App):
    """TrustDNN primary application."""

    class Meta:
        label = 'trustdnn'

        # configuration defaults
        config_defaults = CONFIG

        # call sys.exit() on close
        exit_on_close = True

        # load additional framework extensions
        extensions = [
            'yaml',
            'colorlog',
            'jinja2',
        ]

        # configuration handler
        config_handler = 'yaml'

        # configuration file suffix
        config_file_suffix = '.yml'

        # set the log handler
        log_handler = 'colorlog'

        # set the output handler
        output_handler = 'jinja2'

        interfaces = [
            PluginsInterface, HandlersInterface
        ]

        # register handlers
        handlers = [
            Base, Execute, Evaluate, InstanceHandler
        ]

    def get_plugin_handler(self, name: str, kind: type = None, **kw):
        """
            Gets the handler associated to the plugin

            :param name: label of the plugin
            :param kind: type of the plugin
            :return: handler for the plugin
        """

        try:

            if name not in self.plugin.get_loaded_plugins():
                try:
                    if self.plugin.load_plugin(name):
                        self.log.info(f'Loaded plugin {name}')
                except ModuleNotFoundError as mnf:
                    self.log.error(f'Plugin {name} not found')
                    exit(1)

            plugin = self.handler.resolve('plugins', name)

            if kind is not None:
                if not isinstance(plugin, kind):
                    raise TypeError(f'Plugin {name} is not of type {kind.__name__}')

            if kind == ToolPlugin:
                configs = self.config.get('tools', name)
            elif kind == BenchmarkPlugin:
                configs = self.config.get('benchmarks', name)
            else:
                raise InterfaceError(f'Invalid kind {kind}')

            if configs is None:
                raise KeyError(f'No configuration found for plugin {name}')

            kw.update(configs)
            plugin.__init__(**kw)
            plugin._setup(self)
            self.log.info(f'Initialized plugin {name}')

            return plugin

        except InterfaceError as ie:
            self.log.error(str(ie))
            exit(1)
        except TypeError as te:
            self.log.error(str(te))
            exit(1)
        except KeyError as ke:
            self.log.error(f"Could not resolve plugin {name}")
            exit(1)
        except FileNotFoundError as fnf:
            self.log.error(str(fnf))
            exit(1)

    def load_configs(self):
        import os
        import json

        trustdnn_dir = os.environ.get('TRUSTDNN_DIR', None)

        if trustdnn_dir is None:
            self.log.error('TRUSTDNN_DIR not set')
            exit(1)

        trustdnn_config_path = Path(trustdnn_dir) / 'config'

        if not trustdnn_config_path.exists():
            self.log.error(f'Config path {trustdnn_config_path} not found')
            exit(1)

        tools_config_path = trustdnn_config_path / 'tools'

        if not tools_config_path.exists():
            self.log.error(f'Tools config path {tools_config_path} not found')
            exit(1)

        self.config.add_section('tools')

        for tool_config_path in tools_config_path.iterdir():
            if tool_config_path.is_file() and tool_config_path.suffix == '.json':
                self.log.info(f'Loading tool config {tool_config_path}')
                with tool_config_path.open() as f:
                    self.config.set('tools', tool_config_path.stem, json.load(f))

        benchmarks_config_path = trustdnn_config_path / 'benchmarks'

        if not benchmarks_config_path.exists():
            self.log.error(f'Benchmarks config path {benchmarks_config_path} not found')
            exit(1)

        self.config.add_section('benchmarks')

        for benchmark_config_path in benchmarks_config_path.iterdir():
            if benchmark_config_path.is_file() and benchmark_config_path.suffix == '.json':
                self.log.info(f'Loading benchmark config {benchmark_config_path}')
                with benchmark_config_path.open() as f:
                    self.config.set('benchmarks', benchmark_config_path.stem, json.load(f))


class TrustDNNTest(TestApp, TrustDNN):
    """A sub-class of TrustDNN that is better suited for testing."""

    class Meta:
        label = 'trustdnn'


def main():
    with TrustDNN() as app:
        try:
            app.load_configs()
            app.run()

        except AssertionError as e:
            print('AssertionError > %s' % e.args[0])
            app.exit_code = 1

            if app.debug is True:
                import traceback
                traceback.print_exc()

        except TrustDNNError as e:
            print('TrustDNNError > %s' % e.args[0])
            app.exit_code = 1

            if app.debug is True:
                import traceback
                traceback.print_exc()

        except CaughtSignal as e:
            # Default Cement signals are SIGINT and SIGTERM, exit 0 (non-error)
            print('\n%s' % e)
            app.exit_code = 0


if __name__ == '__main__':
    main()
