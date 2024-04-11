from pathlib import Path
from typing import Dict

from trustdnn.core.split import SplitFactory
from trustdnn.core.split.base import Split


class Dataset:
    def __init__(self, path: Path):
        """
        :param path: Path to the raw dataset
        """

        if not path.exists():
            raise FileNotFoundError(f"Path {path} does not exist.")

        self.name = path.name
        self.path = path

        """
        Loads the dataset from the given path
        """

        self.splits: Dict[str, Split] = {}

        for f in path.iterdir():
            if f.is_dir():
                split = SplitFactory.read_split(f)
                self.splits[split.name] = split

        if len(self.splits) != 3:
            raise ValueError(f"Could not find all splits in {path}")

        if len(set([split.format for split in self.splits.values()])) != 1:
            raise ValueError(f"Splits have different formats")

        self.format = self.splits['train'].format

    @property
    def train(self) -> Split:
        return self.splits['train']

    @property
    def val(self) -> Split:
        return self.splits['val']

    @property
    def test(self) -> Split:
        return self.splits['test']
