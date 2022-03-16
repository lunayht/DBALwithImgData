import numpy as np
import torch

from active_learning import tensor_to_np, select_acq_function
from acquisition_functions import uniform, max_entropy, bald


class TestActiveLearning:
    def test_tensor_to_np(self):
        x = torch.randn((30, 1, 28, 28))
        nx = tensor_to_np(x)
        assert (
            type(nx) == np.ndarray
        ), "expect dtype change from torch.Tensor to np.ndarray"

    def test_select_all_acq_func(self):
        assert select_acq_function(0) == [
            uniform,
            max_entropy,
            bald,
        ], "expect function to query all acq func"

    def test_select_bald_acq_func(self):
        assert select_acq_function(3) == [
            bald
        ], "expect function to query only bald acq func"
