from cement import Handler
from safednn.core.interfaces import PluginsInterface


class PluginHandler(PluginsInterface, Handler):
    class Meta:
        label = 'plugin'

    def __init__(self, name: str = None, **kw):
        super().__init__(**kw)
        self.name = name

    def __str__(self):
        return self.name

