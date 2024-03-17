from pathlib import Path
from safednn.core.split.base import Split, Train, Val, Test


matches = {
    'train': Train,
    'val': Val,
    'test': Test
}


def load(path: Path) -> Split:
    if path.name not in matches:
        raise ValueError(f"Invalid split directory: {path}")

    args = {'_path': path}
    accepted_suffixes = ['csv', 'npy']
    suffixes = []

    for file in path.iterdir():
        if file.suffix.replace('.', '') not in accepted_suffixes:
            continue

        if file.stem == 'x':
            args['_features_file'] = file.name
        elif file.stem == 'y':
            args['_labels_file'] = file.name

        suffixes.append(file.suffix.replace('.', ''))

    if '_features_file' not in args:
        raise ValueError(f"Features file not found in {path}")

    if '_labels_file' not in args:
        raise ValueError(f"Labels file not found in {path}")

    if len(set(suffixes)) != 1:
        raise ValueError(f"All splits must have the same file format.")

    # TODO: define split format

    return matches[path.name](**args)
