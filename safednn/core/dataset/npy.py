import numpy as np
from safednn.core.dataset.base import Dataset


class NPYDataset(Dataset):
    def __init__(self, **kwargs):
        super().__init__(data_format='npy', **kwargs)
        # TODO: update so it does not have to iterate over all files
        #self.path = [file for file in self.root_path.iterdir() if file.is_file() and file.suffix == '.npz'][0]

    @property
    def data(self):
        #if self._data is None:
        #    self._data = np.load(str(self.path))

        return self._data

    @data.setter
    def data(self, data):
        self._data = data
