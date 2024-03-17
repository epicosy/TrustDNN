from safednn.core.dataset.base import Dataset
from safednn.core.dataset.csv import CSVDataset
from safednn.core.dataset.npy import NPYDataset
from safednn.core.split import load as load_split

from pathlib import Path


def load(path: Path) -> Dataset:
    """
    Loads the dataset from the given path
    """

    splits = {}

    for f in path.iterdir():
        if f.is_dir():
            split = load_split(f)
            splits[split.name] = split

    # TODO: check if all splits are present
    if len(splits) != 3:
        raise ValueError(f"Could not find all splits in {path}")

    if len(set([split._format for split in splits.values()])) != 1:
        raise ValueError(f"Splits have different formats")

    # TODO: should blow before reaching this part if the format is not supported

    if splits['train']._format == 'csv':
        return CSVDataset(path=path, splits=splits)
    else:
        return NPYDataset(path=path, splits=splits)
