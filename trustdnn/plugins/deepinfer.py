from pathlib import Path

import pandas as pd

from trustdnn.handlers.tool import ToolPlugin
from trustdnn.core.dataset.base import Dataset
from trustdnn.core.model import Model


class DeepInfer(ToolPlugin):
    class Meta:
        label = 'deepinfer'

    def __init__(self, condition: str = None, prediction_interval: float = None, **kw):
        super().__init__('deepinfer', **kw)
        self.condition = condition
        self.prediction_interval = prediction_interval

    def analyze_command(self, model: Model, dataset: Dataset, working_dir: Path, **kwargs):
        command_args = f"-m {model.path} -wd {working_dir} "

        if self.condition:
            command_args += f"-c \"{self.condition}\" "

        command = f"{command_args} analyze -vx {dataset.val.features_path} "

        if self.prediction_interval:
            command += f"-pi {self.prediction_interval} "

        output = working_dir / 'analysis.json'

        return output, command

    def infer_command(self, model: Model, dataset: Dataset, working_dir: Path, **kwargs):
        command_args = f"-m {model.path} -wd {working_dir} "

        if self.condition:
            command_args += f"-c \"{self.condition}\" "

        subcommand = f"{command_args} infer -tx {dataset.test.features_path}"
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
