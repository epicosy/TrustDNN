
from cement import App, TestApp, init_defaults
from cement.core.exc import CaughtSignal, InterfaceError
from .core.exc import SafeDNNError
from .controllers.base import Base
from .controllers.execute import Execute

from safednn.core.interfaces import PluginsInterface, HandlersInterface
from safednn.handlers.instance import InstanceHandler


# configuration defaults
CONFIG = init_defaults('safednn')
CONFIG['safednn']['foo'] = 'bar'


class SafeDNN(App):
    """SafeDNN primary application."""

    class Meta:
        label = 'safednn'

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
            Base, Execute, InstanceHandler
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


class SafeDNNTest(TestApp,SafeDNN):
    """A sub-class of SafeDNN that is better suited for testing."""

    class Meta:
        label = 'safednn'


def main():
    with SafeDNN() as app:
        try:
            app.run()

        except AssertionError as e:
            print('AssertionError > %s' % e.args[0])
            app.exit_code = 1

            if app.debug is True:
                import traceback
                traceback.print_exc()

        except SafeDNNError as e:
            print('SafeDNNError > %s' % e.args[0])
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
