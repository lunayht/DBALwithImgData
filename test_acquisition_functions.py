import torch
import numpy as np

from acquisition_functions import uniform, max_entropy, bald, var_ratios


class TestAcqFunc:
    def test_uniform_select_2(self):
        x = torch.rand((100, 1, 28, 28))
        idx, chosen_x = uniform(X_pool=x, n_query=2)
        assert len(chosen_x) == 2, "expect 2 points are selected randomly"

    def test_uniform_select_0(self):
        x = torch.rand((100, 1, 28, 28))
        idx, chosen_x = uniform(X_pool=x, n_query=0)
        assert len(chosen_x) == 0, "expect no point is selected randomly"

    def Xtest_max_entropy_select_2(self):
        x = torch.rand((100, 1, 28, 28))
        idx, chosen_x = max_entropy(X_pool=x, n_query=2, T=5)
        assert len(chosen_x) == 2, "expect 2 points are selected from max entropy"

    def Xtest_max_entropy_select_0(self):
        x = torch.rand((100, 1, 28, 28))
        idx, chosen_x = max_entropy(X_pool=x, n_query=0, T=5)
        assert len(chosen_x) == 0, "expect no point is selected from max entropy"
