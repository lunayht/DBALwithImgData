import torch
import numpy as np

from main import tensor_to_np


class TestMain:
    def test_tensor_to_np(self):
        x = torch.randn((30, 1, 28, 28))
        nx = tensor_to_np(x)
        assert type(nx) == np.ndarray, \
            "expect dtype change from torch.Tensor to np.ndarray"