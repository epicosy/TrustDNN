import pandas as pd
from pathlib import Path

from safednn.core.dataset.base import Dataset


class CSVDataset(Dataset):
    def __init__(self, **kwargs):
        super().__init__(data_format='csv', **kwargs)
        # TODO: update so it does not have to iterate over all files
        #self.path = [file for file in self.root_path.iterdir() if file.is_file() and file.suffix == '.csv'][0]

    @property
    def data(self):
        #if self._data is None:
        #    self._data = pd.read_csv(str(self.path), delimiter=',', encoding='utf-8', index_col=False)
            # drop any unnamed index column
        #    self._data = self._data.loc[:, ~self._data.columns.str.contains('^Unnamed')]

        return self._data

    @data.setter
    def data(self, data):
        self._data = data
