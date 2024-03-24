import numpy as np

from safednn.core.split.base import Split


class NPYSplit(Split):
    format = 'npy'

    @property
    def features(self):
        if self._features is None:
            self._features = np.load(self.features_path)

        return self._features

    @property
    def labels(self):
        if self._labels is None:
            self._labels = np.load(self.labels_path)

        return self._labels

    def save(self):
        self.path.mkdir(parents=True, exist_ok=True)
        np.save(self.features_path, self._features)
        np.save(self.labels_path, self._labels)
