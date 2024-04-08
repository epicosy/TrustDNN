from pathlib import Path

import pandas as pd

from safednn.handlers.tool import ToolPlugin
from safednn.core.dataset.base import Dataset
from safednn.core.model import Model


class DeepInfer(ToolPlugin):
    class Meta:
        label = 'deepinfer'

    def __init__(self, **kw):
        super().__init__('deepinfer', **kw)

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
        violations_path = Path(output).parent / 'violations.csv'
        satisfactions_path = Path(output).parent / 'satisfactions.csv'
        violations = None
        satisfactions = None

        if violations_path.exists():
            violations = pd.read_csv(violations_path)

        if satisfactions_path.exists():
            satisfactions = pd.read_csv(satisfactions_path)

        print('Model:', Path(output).parent.name, 'Violations:', violations['0'].sum(), 'Satisfactions:',
              satisfactions['0'].sum())

        # replace all 'wrong with incorrect'

        df['implication'] = df['implication'].str.replace('Wrong', 'incorrect')
        df['implication'] = df['implication'].str.replace('Correct', 'correct')

        return df.rename(columns={'implication': 'notification'})


def load(app):
    app.handler.register(DeepInfer)
