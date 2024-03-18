from cement import Handler, Interface


class PluginsInterface(Interface):
    """
        Handlers' Interface
    """
    class Meta:
        """
            Meta class
        """
        interface = 'plugins'


class PluginHandler(PluginsInterface, Handler):
    class Meta:
        label = 'plugin'

    def __init__(self, name: str = None, **kw):
        super().__init__(**kw)
        self.name = name

    def __str__(self):
        return self.name

