from abc import abstractmethod
from safednn.handlers.plugin import PluginHandler


class ToolPlugin(PluginHandler):
    class Meta:
        label = 'tool'

    def __init__(self, name: str, **kw):
        super().__init__(name, **kw)

    @abstractmethod
    def run(self, **kwargs):
        """
            Run the tool
        :param kwargs:
        :return:
        """
        pass

    def __str__(self):
        return self.help()

    @abstractmethod
    def help(self):
        """
            Return the help message
        :return:
        """
        pass
