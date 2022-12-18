
__all__ = ['InceptionModule', 'InceptionBlock', 'InceptionTime']

import torch
from torch import nn
from utils import *
from layers import *
from typing import cast, Union, List
from torchsummaryX import summary


class InceptionTime(nn.Module):
    """Official implementation in Keras framework:
    https://github.com/hfawaz/InceptionTime/blob/master/classifiers/inception.py.
    Original paper:
    https://arxiv.org/pdf/1909.04939.pdf

    Attributes
    ----------
    in_channels: int
        The number of input channels (i.e. input.shape[-1])
    out_channels: int
        The number of "hidden channels" to use. Can be a list (for each block) or an
        int, in which case the same value will be applied to each block
    use_bottleneck: bool
        Boolean indicating whether to use bottleneck layers or not
    kernel_size: int
        The size of the kernels to use for each inception block. Within each block, each
        of the 3 convolutional layers will have kernel size
        `[kernel_size // (2 ** i) for i in range(3)]`
    use_residuals: bool
        Boolean indicating whether to use residual connections or not
    depth: int
        The number of inception modules
    num_pred_classes:
        The number of output classes
    """

    def __init__(self, in_channels: int = 8, out_channels: int = 32, use_bottleneck: bool = True,
                 kernel_size: int = 64, use_residuals: bool = True, depth: int = 3, num_pred_classes: int = 3):

        super().__init__()

        self.input_args = {'in_channels': in_channels,
                           'out_channels': out_channels,
                           'use_bottleneck': use_bottleneck,
                           'kernel_size': kernel_size,
                           'use_residuals': use_residuals,
                           'depth': depth,
                           'num_pred_classes': num_pred_classes
        }

        self.inceptionblock = InceptionBlock(in_channels, out_channels, use_bottleneck,
                                             kernel_size, use_residuals, depth)
        self.gap = GAP1d(1, out_channels)
        self.linear = nn.Linear(out_channels * 4, num_pred_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        #x = x[:, :, 0::4]
        x = self.inceptionblock(x)#.mean(dim=-1)  # .mean(dim=-1) corresponds to GAP
        x = self.gap(x)
        return self.linear(x)


class InceptionBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: Union[List[int], int], use_bottleneck: bool = True,
                 kernel_size: int = 128, use_residuals: bool = True, depth: int = 6, stride: int = 1) -> None:
        super().__init__()

        self.use_residual, self.depth = use_residuals, depth
        self.inception, self.shortcut = nn.ModuleList(), nn.ModuleList()
        for d in range(depth):
            self.inception.append(InceptionModule(in_channels if d == 0 else out_channels * 4, out_channels, stride,
                                                  use_bottleneck, kernel_size))
            if self.use_residual and d % 3 == 2:
                n_in, n_out = in_channels if d == 2 else out_channels * 4, out_channels * 4
                self.shortcut.append(nn.BatchNorm1d(n_in) if n_in == n_out else ConvBlock(n_in, n_out, kernel_size=1,
                                                                                          stride=1, activation=None))

        self.activation = nn.ReLU()

    def forward(self, x):
        #x = x[:, :, 0::4]
        residual = x
        for d, l in enumerate(range(self.depth)):
            x = self.inception[d](x)
            if self.use_residual and d % 3 == 2:
                residual = x = self.activation(x.add(self.shortcut[d // 3](residual)))
        return x


class InceptionModule(nn.Module):
    """An inception module consists of an (optional) bottleneck, followed
    by 3 conv1d layers.
    """

    def __init__(self, in_channels: int, out_channels: int,
                 stride: int = 1, use_bottleneck: bool = True,
                 kernel_size: int = 128) -> None:
        assert kernel_size > 3, "Kernel size must be strictly greater than 3"
        super().__init__()

        self.use_bottleneck = use_bottleneck if in_channels > 1 else False
        if self.use_bottleneck:
            self.bottleneck = conv1d_same_padding(in_channels, out_channels, 1)

        kernel_size_s = [kernel_size // (2 ** i) for i in range(3)]
        kernel_size_s = [k if k % 2 != 0 else k - 1 for k in kernel_size_s]  # ensure odd ks

        self.conv_layers = nn.ModuleList([conv1d_same_padding(out_channels if self.use_bottleneck else in_channels,
                                                              out_channels, k, stride=stride) for k in kernel_size_s])

        self.max_pool_conv = nn.Sequential(*[nn.MaxPool1d(3, stride=1, padding=1),
                                             conv1d_same_padding(in_channels, out_channels, 1)])

        self.batch_norm = nn.BatchNorm1d(num_features=out_channels*4)
        self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore
        org_x = x
        if self.use_bottleneck:
            x = self.bottleneck(org_x)
        x = torch.cat([layer(x) for layer in self.conv_layers] + [self.max_pool_conv(org_x)], dim=1)
        return self.activation(self.batch_norm(x))


if __name__ == "__main__":
    batch_size = 16
    num_features = 8
    window_len = 4096
    num_classes = 3
    xb = torch.rand(batch_size, num_features, window_len)
    model = InceptionTime(in_channels=8, num_pred_classes=3)
    print('\nAsserting...')
    assert InceptionTime(num_features, num_pred_classes=num_classes)(xb).shape == (batch_size, num_classes)

    assert InceptionTime(num_features, num_pred_classes=num_classes, use_bottleneck=False)(xb).shape == \
           (batch_size, num_classes)

    assert InceptionTime(num_features, num_pred_classes=num_classes, use_residuals=False)(xb).shape == \
           (batch_size, num_classes)

    #assert count_parameters(InceptionTime(3, 2)) == 455490
    print('\nDone asserting!')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = InceptionTime(in_channels=8, out_channels=16, kernel_size=256, depth=6, num_pred_classes=3)
    model.to(device)
    print('\n\nPrinting model summary')
    #print(summary(model, (3, 12)))
    print(summary(model, torch.rand([1, 8, 4096]).to(device)))
    #print(count_parameters(InceptionTime(8, 3)))
    #print('\n\nPrinting model')
    #print(model)
    output = model(torch.randn(10, 8, 4096, device=device))

