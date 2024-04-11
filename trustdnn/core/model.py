import pandas as pd
from pathlib import Path


class Model:
    def __init__(self, path: Path, dataset: str, predictions_path: Path):
        if not path.exists():
            raise ValueError(f"Model file {path} does not exist")

        self.name: str = path.stem
        self.dataset: str = dataset
        self.path: Path = path

        # look for predictions in predictions directory
        self.predictions_path: Path = predictions_path

        if not self.predictions_path.exists():
            raise ValueError(f"Predictions file {self.predictions_path} does not exist")

        self._predictions = None

    @property
    def predictions(self) -> pd.DataFrame:
        if self._predictions is None:
            self._predictions = pd.read_csv(self.predictions_path)

        return self._predictions
