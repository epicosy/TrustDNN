from pathlib import Path

from safednn.handlers.tool import ToolPlugin
from safednn.core.dataset.base import Dataset
from safednn.core.model import Model


class Prophecy(ToolPlugin):
    class Meta:
        label = 'prophecy'

    def __init__(self, **kw):
        super().__init__('prophecy', command='prophecy.main', path='~/projects/ProphecyPlus',
                         interpreter="python3 -m", env_path='~/projects/ProphecyPlus/env', **kw)

    def analyze_command(self, model: Model, dataset: Dataset, working_dir: Path, **kwargs):
        command = f"-m {model.path} -wd {working_dir} analyze "
        command += f"-tx {dataset.train.features_path} -ty {dataset.train.labels_path} "
        command += f"-vx {dataset.val.features_path} -vy {dataset.val.labels_path}"
        output = working_dir / 'ruleset.csv'

        return output, command

    def infer_command(self, model: Model, dataset: Dataset, working_dir: Path, **kwargs):
        subcommand = f"-m {model.path} -wd {working_dir} infer "
        subcommand += f"-tx {dataset.test.features_path} -ty {dataset.test.labels_path} classifiers"
        output = working_dir / 'predictions' / 'results_clf.csv'

        return output, subcommand


def load(app):
    app.handler.register(Prophecy)
