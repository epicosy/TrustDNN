from safednn.handlers.benchmark import BenchmarkPlugin


class TrustBench(BenchmarkPlugin):
    class Meta:
        label = 'trustbench'

    def __init__(self, **kw):
        super().__init__(name='trustbench', **kw)

    def help(self):
        pass


def load(app):
    app.handler.register(TrustBench)
