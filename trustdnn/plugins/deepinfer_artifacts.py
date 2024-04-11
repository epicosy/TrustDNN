from trustdnn.handlers.benchmark import BenchmarkPlugin


class DeepInferArtifacts(BenchmarkPlugin):
    class Meta:
        label = 'deepinfer_artifacts'

    def __init__(self, **kw):
        super().__init__(name='deepinfer_artifacts', **kw)

    def help(self):
        pass


def load(app):
    app.handler.register(DeepInferArtifacts)
