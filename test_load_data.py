import torch
import numpy as np
import load_data


class TestLoadData:
    def test_preprocess_training_data(self):
        train_data, test_data = load_data.download_dataset()
        x, y, _, _, _, _, _, _ = load_data.split_and_load_dataset(train_data, test_data)
        x_init, y_init = load_data.preprocess_training_data(x, y)
        assert x_init.shape[0] == 20, "expect 20 initial datapoints"
        assert np.all(np.bincount(y_init) == 2), "expect balanced distribution"
