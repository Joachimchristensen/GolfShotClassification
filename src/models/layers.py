
__all__ = ['same_padding_1d', 'Conv1dSamePadding', 'conv1d_same_padding', 'ConvBlock', 'GAP1d', 'Flatten']

from torch import nn


def same_padding_1d(seq_len, kernel_size, stride=1, dilation=1):
    """Same padding formula as used in Tensorflow"""
    p = (seq_len - 1) * stride + (kernel_size - 1) * dilation + 1 - seq_len
    return p // 2, p - p // 2


class Conv1dSamePadding(nn.Module):
    """Conv1d with padding='same'"""

    def __init__(self, input_channels, out_channels, ks=3, stride=1, dilation=1):
        super().__init__()
        self.kernel_size, self.stride, self.dilation = ks, stride, dilation
        self.conv1d_same = nn.Conv1d(input_channels, out_channels, ks, stride=stride, dilation=dilation)
        self.weight = self.conv1d_same.weight
        self.bias = self.conv1d_same.bias
        self.pad = nn.ConstantPad1d

    def forward(self, x):
        padding = same_padding_1d(x.shape[-1], self.kernel_size, dilation=self.dilation)
        return self.conv1d_same(self.pad(padding, value=0)(x))


def conv1d_same_padding(input_channels, out_channels, kernel_size=None, ks=None, stride=1, dilation=1):
    """conv1d layer with padding='same'"""
    assert not (kernel_size and ks), 'use kernel_size or ks but not both simultaneously'
    assert kernel_size is not None or ks is not None, 'you need to pass a ks'
    kernel_size = kernel_size or ks
    if kernel_size % 2 == 1:
        conv = nn.Conv1d(input_channels, out_channels, kernel_size, stride=stride, padding=kernel_size//2 * dilation,
                         dilation=dilation)
    else:
        conv = Conv1dSamePadding(input_channels, out_channels, kernel_size, stride=stride, dilation=dilation)

    return conv


class ConvBlock(nn.Sequential):
    """Create a sequence of conv1d_same_padding (`ni` to `nf`), activation (if `act_cls`) and `norm_type` layers."""

    def __init__(self, input_channels, out_channels, kernel_size=None, ks=3, stride=1, activation=nn.ReLU, dropout=0.2):
        kernel_size = kernel_size or ks
        layers = []
        conv = conv1d_same_padding(input_channels, out_channels, ks=kernel_size, stride=stride)
        activation = None if activation is None else activation()
        layers += [conv]
        activation_bn = []
        if activation is not None:
            activation_bn.append(activation)
        activation_bn.append(nn.BatchNorm1d(out_channels))
        if dropout:
            layers += [nn.Dropout(dropout)]
        layers += activation_bn
        super().__init__(*layers)


class GAP1d(nn.Module):
    """Global Adaptive Pooling + Flatten"""
    def __init__(self, output_size=1, out_channels=32):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool1d(output_size)
        self.flatten = Flatten(out_channels*4*1)

    def forward(self, x):
        return self.flatten(self.gap(x))


class Flatten(nn.Module):
    def __init__(self, out_features):
        super().__init__()
        self.output_dim = out_features

    def forward(self, x):
        return x.view(-1, self.output_dim)


