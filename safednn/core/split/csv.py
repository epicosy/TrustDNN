import numpy as np
import pandas as pd

from safednn.core.split import Split


class CSVSplit(Split):
    format = 'csv'

    def features(self):
        if self._features is None:
            self._features = pd.read_csv(self.features_path, delimiter=',', encoding='utf-8',
                                         header=None if not self.headers else 'infer')

        return self._features

    def labels(self):
        if self._labels is None:
            # TODO: should be pandas dataframe
            self._labels = np.loadtxt(self.labels_path, dtype=int)

        return self._labels

    def save(self):
        self.path.mkdir(parents=True, exist_ok=True)
        self._features.to_csv(self.features_path, index=False, header=self.headers)
        # TODO: should be pandas dataframe
        np.savetxt(self.labels_path, self._labels, fmt='%d')
