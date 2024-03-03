from safednn.handlers.benchmark import BenchmarkPlugin


class TrustBench(BenchmarkPlugin):
    class Meta:
        label = 'trustbench'

    def __init__(self, **kw):
        super().__init__(name='trustbench', **kw)

    def models(self):
        return []

    def datasets(self):
        return []

    def get_dataset(self, name: str):
        return None

    def get_model(self, name: str):
        return None


def load(app):
    app.handler.register(TrustBench)
