import torch

from cnn import CNN


class TestCNN:
    # def test_conv_relu_output_size(self):
    #     net = CNN()
    #     assert net(torch.randn(1, 1, 28, 28)).size() == (1, 32, 22, 22), \
    #         "expect output size of (1, 32, 22, 22)"
    # def test_maxpool2d_output_size(self):
    #     net = CNN()
    #     assert net(torch.randn(1, 1, 28, 28)).size() == (1, 32, 11, 11), \
    #         "expect output size of (1, 32, 11, 11)"
    # def test_linear_output_size(self):
    #     net = CNN()
    #     assert net(torch.randn(1, 1, 28, 28)).size() == (1, 3872), \
    #         "expect output size of (1, 3872)"
    def test_output_size_28x28(self):
        net = CNN()
        assert net(torch.randn(1, 1, 28, 28)).size() == (
            1,
            10,
        ), "expect output size of (1,10)"

    def test_output_size_32x32(self):
        net = CNN(img_rows=32, img_cols=32)
        assert net(torch.randn(1, 1, 32, 32)).size() == (
            1,
            10,
        ), "expect output size of (1, 10)"
