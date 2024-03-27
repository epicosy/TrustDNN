import numpy as np
import pandas as pd

from pathlib import Path

from safednn.handlers.tool import ToolPlugin
from safednn.core.dataset.base import Dataset
from safednn.core.model import Model


class SelfChecker(ToolPlugin):
    class Meta:
        label = 'selfchecker'

    def __init__(self, **kw):
        super().__init__('selfchecker', command='selfchecker.main', path='~/projects/nasa/SelfCheckerPlus',
                         interpreter="python3 -m", env_path='~/projects/nasa/SelfCheckerPlus/env', **kw)

    def analyze_command(self, model: Model, dataset: Dataset, working_dir: Path, **kwargs):
        command = f"-m {model.path} -wd {working_dir} analyze "
        command += f"-tx {dataset.train.features_path} -ty {dataset.train.labels_path} "
        command += f"-vx {dataset.val.features_path} -vy {dataset.val.labels_path}"
        output = working_dir / 'pred_labels_valid.npy'

        return output, command

    def infer_command(self, model: Model, dataset: Dataset, working_dir: Path, **kwargs):
        subcommand = f"-m {model.path} -wd {working_dir} infer "
        subcommand += f"-tx {dataset.test.features_path} -ty {dataset.test.labels_path}"
        output = working_dir / 'pred_labels_test.npy'

        return output, subcommand

    def get_notifications(self, output: Path, **kwargs):
        pred_labels = np.load(output).astype(int)
        notifications = []

        for idx, _ in enumerate(pred_labels):
            if pred_labels[idx].T[-2] != pred_labels[idx].T[-1]:
                notifications.append('incorrect')
            else:
                notifications.append('correct')

        return pd.DataFrame(notifications, columns=['notification'])


def load(app):
    app.handler.register(SelfChecker)
