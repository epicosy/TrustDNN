from safednn.handlers.tool import ToolPlugin
from safednn.core.dataset.base import Dataset


class Prophecy(ToolPlugin):
    class Meta:
        label = 'prophecy'

    def __init__(self, **kw):
        super().__init__('prophecy', command='prophecy.main', path='~/projects/ProphecyPlus',
                         env_path='~/projects/ProphecyPlus/env', **kw)

    def run(self, model: str, dataset: Dataset, **kwargs):
        subcommand = f"--model {model} detect -tx {dataset.test.features_path} -ty {dataset.test.labels_path}"
        self.run_command(subcommand, interpreter="python3 -m", stdout=True)

    def get_results(self, **kwargs):
        # TODO: to be implemented
        pass

    def help(self):
        self.run_command('--help', interpreter="python3 -m", stdout=True)


def load(app):
    app.handler.register(Prophecy)
