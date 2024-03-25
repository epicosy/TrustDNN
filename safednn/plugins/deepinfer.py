from pathlib import Path

import pandas as pd

from safednn.handlers.tool import ToolPlugin
from safednn.core.dataset.base import Dataset
from safednn.core.model import Model


class DeepInfer(ToolPlugin):
    class Meta:
        label = 'deepinfer'

    def __init__(self, **kw):
        super().__init__('deepinfer', command='deepinfer.main', path='~/projects/nasa/DeepInferPlus',
                         interpreter="python3 -m", env_path='~/projects/nasa/DeepInferPlus/env', **kw)

    def analyze_command(self, model: Model, dataset: Dataset, working_dir: Path, **kwargs):
        command = f"-m {model.path} -wd {working_dir} analyze "
        command += f"-vx {dataset.val.features_path} "
        output = working_dir / 'analysis.json'

        return output, command

    def infer_command(self, model: Model, dataset: Dataset, working_dir: Path, **kwargs):
        subcommand = f"-m {model.path} -wd {working_dir} infer "
        subcommand += f"-tx {dataset.test.features_path}"
        output = working_dir / 'implications.csv'

        return output, subcommand

    def get_notifications(self, output: Path, **kwargs):
        df = pd.read_csv(output)
        # replace all 'wrong with incorrect'

        df['implication'] = df['implication'].str.replace('Wrong', 'incorrect')

        return df.rename(columns={'implication': 'notification'})


def load(app):
    app.handler.register(DeepInfer)
