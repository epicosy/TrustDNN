import pandas as pd
from pathlib import Path

from trustdnn.handlers.tool import ToolPlugin
from trustdnn.core.dataset.base import Dataset
from trustdnn.core.model import Model


class Prophecy(ToolPlugin):
    class Meta:
        label = 'prophecy'

    def __init__(self, only_dense_layers: bool = False, only_activation_layers: bool = False, skip_rules: bool = False,
                 random_state: int = None, balance: bool = False, **kw):
        super().__init__('prophecy', **kw)
        self.only_dense_layers = only_dense_layers
        self.only_activation_layers = only_activation_layers
        self.skip_rules = skip_rules
        self.random_state = random_state
        self.balance = balance

    def analyze_command(self, model: Model, dataset: Dataset, working_dir: Path, **kwargs):
        command = f"-m {model.path} -wd {working_dir} analyze "
        command += (f"-tx {dataset.train.features_path} -ty {dataset.train.labels_path} "
                    f"-vx {dataset.val.features_path} -vy {dataset.val.labels_path} ")

        if self.only_dense_layers:
            command += "-odl "

        if self.only_activation_layers:
            command += "-oal "

        if self.skip_rules:
            command += "-sr "

        if self.random_state:
            command += f"-rs {self.random_state} "

        if self.balance:
            command += "-b "

        output = working_dir / 'ruleset.csv'

        return output, command

    def infer_command(self, model: Model, dataset: Dataset, working_dir: Path, **kwargs):
        subcommand = f"-m {model.path} -wd {working_dir} infer "
        subcommand += f"-tx {dataset.test.features_path} -ty {dataset.test.labels_path} classifiers"
        output = working_dir / 'predictions' / 'results_clf.csv'

        return output, subcommand

    # TODO: there is a better way of doing this and make it part of the plugin, just provide output path and column name
    def get_notifications(self, output: Path, **kwargs):
        df = pd.read_csv(output)

        return df.rename(columns={'outcome': 'notification'})


def load(app):
    app.handler.register(Prophecy)
