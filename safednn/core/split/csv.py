import numpy as np
import pandas as pd

from safednn.core.split import Split


class CSVSplit(Split):
    format = 'csv'

    @property
    def features(self) -> pd.DataFrame:
        if self._features is None:
            self._features = pd.read_csv(self.features_path, delimiter=',', encoding='utf-8',
                                         header=None if not self.headers else 'infer')

        return self._features

    @property
    def labels(self) -> pd.DataFrame:
        if self._labels is None:
            self._labels = pd.read_csv(self.labels_path, delimiter=',', dtype=np.int32, encoding='utf-8',
                                       header=None if not self.headers else 'infer')

        return self._labels

    def save(self):
        self.path.mkdir(parents=True, exist_ok=True)
        self._features.to_csv(self.features_path, index=False, header=self.headers)
        # TODO: should be pandas dataframe
        np.savetxt(self.labels_path, self._labels, fmt='%d')
