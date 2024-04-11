import json

import numpy as np
import pandas as pd

from pathlib import Path

from trustdnn.handlers.tool import ToolPlugin
from trustdnn.core.dataset.base import Dataset
from trustdnn.core.model import Model


class SelfChecker(ToolPlugin):
    class Meta:
        label = 'selfchecker'

    def __init__(self, batch_size: int = 128, only_dense_layers: bool = False, only_activation_layers: bool = False,
                 var_threshold: float = None, **kw):
        super().__init__('selfchecker', **kw)
        self.batch_size = batch_size
        self.only_activation_layers = only_activation_layers
        self.only_dense_layers = only_dense_layers
        self.has_metrics = True
        self.var_threshold = var_threshold

    def analyze_command(self, model: Model, dataset: Dataset, working_dir: Path, **kwargs):
        command_args = f"-m {model.path} -wd {working_dir} -bs {self.batch_size} "

        if self.only_dense_layers:
            command_args += "-odl "

        if self.only_activation_layers:
            command_args += "-oal "

        command = f"{command_args} analyze -tx {dataset.train.features_path} -ty {dataset.train.labels_path} "
        command += f"-vx {dataset.val.features_path} -vy {dataset.val.labels_path}"

        if self.var_threshold:
            command += f" --var_threshold {self.var_threshold}"

        output = working_dir / 'pred_labels_valid.npy'

        return output, command

    def infer_command(self, model: Model, dataset: Dataset, working_dir: Path, **kwargs):
        command_args = f"-m {model.path} -wd {working_dir} -bs {self.batch_size} "

        if self.only_dense_layers:
            command_args += "-odl "

        if self.only_activation_layers:
            command_args += "-oal "

        subcommand = f"{command_args} infer -tx {dataset.test.features_path} -ty {dataset.test.labels_path}"
        output = working_dir / 'performance.json'

        return output, subcommand

    # This is just a shortcut to get the metrics for selfchecker, but a tool is supposed to return the notifications
    @staticmethod
    def get_metrics(output: Path):
        with open(output, 'r') as f:
            return json.load(f)

    def get_notifications(self, output: Path, **kwargs):
        pass

        #pred_labels = np.load(output).astype(int)
        #notifications = []

        #for idx, _ in enumerate(pred_labels):
        #    if pred_labels[idx].T[-2] != pred_labels[idx].T[-1]:
        #        notifications.append('incorrect')
        #    else:
        #        notifications.append('correct')

        #return pd.DataFrame(notifications, columns=['notification'])


def load(app):
    app.handler.register(SelfChecker)
