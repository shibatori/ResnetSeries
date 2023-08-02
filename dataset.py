import random
import numpy as np
import mindspore.dataset as ds
from mindspore.dataset import vision, transforms

# Iterable dataset as iterable input.

class MyIterable_test():
    def __init__(self):
        dataset_org = np.load('UNSWNB_labelall_test_dataset.npz')
        data = dataset_org['data']
        data = np.resize(data, (data.shape[0], 4, 4, 4))
        data_32 = data.astype(np.float32)
        label = dataset_org['label']
        label = np.resize(label, (label.shape[0]))

        print(data.shape)
        print(label.shape)
        self._index = 0
        self._data = data_32
        self._label = label

    def __next__(self):
        if self._index >= len(self._data):
            raise StopIteration
        else:
            item = (self._data[self._index], self._label[self._index])
            self._index += 1
            return item

    def __iter__(self):
        self._index = 0
        return self

    def __len__(self):
        return len(self._data)

class MyIterable_train():
    def __init__(self):
        dataset_org = np.load('UNSWNB_labelall_train_dataset.npz')
        data = dataset_org['data']
        data = np.resize(data, (data.shape[0], 4, 4, 4))
        data_32 = data.astype(np.float32)
        label = dataset_org['label']
        label = np.resize(label, (label.shape[0]))

        print(data.shape)
        print(label.shape)
        self._index = 0
        self._data = data_32
        self._label = label

    def __next__(self):
        if self._index >= len(self._data):
            raise StopIteration
        else:
            item = (self._data[self._index], self._label[self._index])
            self._index += 1
            return item

    def __iter__(self):
        self._index = 0
        return self

    def __len__(self):
        return len(self._data)

dataset_train = ds.GeneratorDataset(source=MyIterable_train(), column_names=["data", "label"])
dataset_test = ds.GeneratorDataset(source=MyIterable_test(), column_names=["data", "label"])
