from safednn.handlers.tool import ToolPlugin


class Prophecy(ToolPlugin):
    class Meta:
        label = 'prophecy'

    def __init__(self, **kw):
        super().__init__('prophecy', **kw)

    def run(self):
        pass

    def help(self):
        pass


def load(app):
    app.handler.register(Prophecy)
