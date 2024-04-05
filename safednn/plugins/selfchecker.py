import json

import numpy as np
import pandas as pd

from pathlib import Path

from safednn.handlers.tool import ToolPlugin
from safednn.core.dataset.base import Dataset
from safednn.core.model import Model


class SelfChecker(ToolPlugin):
    class Meta:
        label = 'selfchecker'

    def __init__(self, batch_size: int = 128, **kw):
        super().__init__('selfchecker', **kw)
        self.batch_size = batch_size
        self.has_metrics = True

    def analyze_command(self, model: Model, dataset: Dataset, working_dir: Path, **kwargs):
        command = f"-m {model.path} -wd {working_dir} -bs {self.batch_size} analyze "
        command += f"-tx {dataset.train.features_path} -ty {dataset.train.labels_path} "
        command += f"-vx {dataset.val.features_path} -vy {dataset.val.labels_path}"
        output = working_dir / 'pred_labels_valid.npy'

        return output, command

    def infer_command(self, model: Model, dataset: Dataset, working_dir: Path, **kwargs):
        subcommand = f"-m {model.path} -wd {working_dir} -bs {self.batch_size} infer "
        subcommand += f"-tx {dataset.test.features_path} -ty {dataset.test.labels_path}"
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
