from pathlib import Path
from safednn.core.split.base import Split, SPLIT_NAMES

from safednn.core.split.csv import CSVSplit
from safednn.core.split.npy import NPYSplit


SPLIT_FORMATS = {'csv': CSVSplit, 'npy': NPYSplit}


class SplitFactory:
    @staticmethod
    def read_split(path: Path, headers: bool = True) -> Split:
        if path.name not in SPLIT_NAMES:
            raise ValueError(f"Invalid split directory: {path}")

        args = {'path': path, 'name': path.name, 'headers': headers}
        suffixes = []

        has_features_file = False
        has_labels_file = False

        for file in path.iterdir():
            suffix = file.suffix.replace('.', '')

            if suffix not in SPLIT_FORMATS:
                continue

            suffixes.append(suffix)

            if file.stem == 'x':
                has_features_file = True
            elif file.stem == 'y':
                has_labels_file = True

        if not has_features_file:
            raise ValueError(f"Features file not found in {path}")

        if not has_labels_file:
            raise ValueError(f"Labels file not found in {path}")

        formats = set(suffixes)

        if len(formats) != 1:
            raise ValueError(f"All splits must have the same file format.")

        split_format = formats.pop()
        split = SPLIT_FORMATS[split_format]
        args['format'] = split_format

        return split(**args)
