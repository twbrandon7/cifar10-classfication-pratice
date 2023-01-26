"""Implement ResNet [1].

In the first convolution layer of Pytorch's built in ResNet model [2], the kernel size
is 7. However to apply ResNet in smaller image, the kernel size should be reduced. For
example, when classifying images in CIFAR-10, kernel size 3 is used in the first
convolution layer [3].

To change the kernel size, ResNet is re-implemented in this module.


Reference:
[1] K. He, X. Zhang, S. Ren and J. Sun, "Deep Residual Learning for Image Recognition",
    arXiv preprint arXiv:1512.03385, 2015.
[2] "torchvision.models.resnet - Torchvision main documentation," Pytorch,
    https://pytorch.org/vision/main/_modules/torchvision/models/resnet.html (accessed
    Jun. 26, 2023).
[3] "tensorpack/cifar10-resnet.py at master · tensorpack/tensorpack," GitHub,
    https://github.com/tensorpack/tensorpack/blob/master/examples/ResNet/cifar10-resnet.py#L71
    (accessed Jun. 26, 2023).
"""
import torch
from torch import nn


class ResBlock(nn.Module):
    """A residual block."""

    def __init__(self, in_channels: int, out_channels: int, stride: int) -> None:
        """Create a residual block.

        Args:
            in_channels (int): Input channels of the residual block.
            out_channels (int): Output channels of the residual block.
            stride (int): Stride of the first convolution layer in the residual block.
        """
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        if stride != 1 or in_channels != out_channels:
            self.down_sample = nn.Sequential(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.down_sample = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Define how the model is going to run.

        Args:
            x (torch.Tensor): The input tensor. The shape of the tensor should be
            (batch, input channels, image height, image width).

        Returns:
            torch.Tensor: A tensor with shape
            (batch, input channels, image height / stride, image width / stride).
        """
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.down_sample is not None:
            identity = self.down_sample(x)

        out += identity
        out = self.relu(out)

        return out

    @staticmethod
    def make_layer(
        blocks: int, in_channels: int, out_channels: int, stride: int = 1
    ) -> nn.Sequential:
        """Create a stack of residual blocks.

        Args:
            blocks (int): The number of residual blocks.
            in_channels (int): The input channels.
            out_channels (int): The output channels.
            stride (int, optional): Stride of the first convolution layer.
            Defaults to 1.

        Raises:
            Exception: Raise exception if argument `blocks` is smaller than 1.

        Returns:
            nn.Sequential: A stack of residual blocks.
        """
        if blocks < 1:
            raise Exception("Argument `blocks` should equal or greater than 1.")

        list_layers = [
            ResBlock(in_channels=in_channels, out_channels=out_channels, stride=stride)
        ]
        for _ in range(1, blocks):
            list_layers.append(
                ResBlock(in_channels=out_channels, out_channels=out_channels, stride=1)
            )
        return nn.Sequential(*list_layers)


class ResNet(nn.Module):
    """A residual network."""

    def __init__(
        self,
        blocks_layer1: int,
        blocks_layer2: int,
        blocks_layer3: int,
        blocks_layer4: int,
        num_classes: int = 10,
        init_kernel_size: int = 7,
    ) -> None:
        """Create a residual network.

        Args:
            blocks_layer1 (int): The number of residual blocks with channel counts 64.
            blocks_layer2 (int): The number of residual blocks with channel counts 128.
            blocks_layer3 (int): The number of residual blocks with channel counts 256.
            blocks_layer4 (int): The number of residual blocks with channel counts 512.
            num_classes (int, optional): The number of the classes to classify.
            Defaults to 10.
            init_kernel_size (int, optional): The kernel size of the first convolution
            layer. Defaults to 7.
        """
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=64,
            kernel_size=init_kernel_size,
            stride=2,
            padding=3,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(num_features=64)
        self.relu = nn.ReLU(inplace=True)
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = ResBlock.make_layer(
            blocks=blocks_layer1, in_channels=64, out_channels=64, stride=1
        )
        self.layer2 = ResBlock.make_layer(
            blocks=blocks_layer2, in_channels=64, out_channels=128, stride=2
        )
        self.layer3 = ResBlock.make_layer(
            blocks=blocks_layer3, in_channels=128, out_channels=256, stride=2
        )
        self.layer4 = ResBlock.make_layer(
            blocks=blocks_layer4, in_channels=256, out_channels=512, stride=2
        )

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

        # initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Define how the model is going to run.

        Args:
            x (torch.Tensor): The input tensor. The shape of the tensor should be
            (batch, input channels, image height, image width).

        Returns:
            torch.Tensor: The classification result in on-hot encoding.
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.max_pool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

    @staticmethod
    def make_resnet18(num_classes: int, init_kernel_size: int = 7):
        """Create a ResNet-18 model.

        Args:
            num_classes (int): The number of classes to classify.
            init_kernel_size (int, optional): The number of the classes to classify.
            Defaults to 7.

        Returns:
            _type_: A ResNet-18 model.
        """
        return ResNet(
            2, 2, 2, 2, num_classes=num_classes, init_kernel_size=init_kernel_size
        )

    @staticmethod
    def make_resnet34(num_classes: int, init_kernel_size: int = 7):
        """Create a ResNet-34 model.

        Args:
            num_classes (int): The number of classes to classify.
            init_kernel_size (int, optional): The number of the classes to classify.
            Defaults to 7.

        Returns:
            _type_: A ResNet-34 model.
        """
        return ResNet(
            3, 4, 6, 3, num_classes=num_classes, init_kernel_size=init_kernel_size
        )

    @staticmethod
    def make_resnet50(num_classes: int, init_kernel_size: int = 7):
        """Create a ResNet-50 model.

        Args:
            num_classes (int): The number of classes to classify.
            init_kernel_size (int, optional): The number of the classes to classify.
            Defaults to 7.

        Returns:
            _type_: A ResNet-50 model.
        """
        return ResNet(
            3, 4, 6, 3, num_classes=num_classes, init_kernel_size=init_kernel_size
        )

    @staticmethod
    def make_resnet101(num_classes: int, init_kernel_size: int = 7):
        """Create a ResNet-101 model.

        Args:
            num_classes (int): The number of classes to classify.
            init_kernel_size (int, optional): The number of the classes to classify.
            Defaults to 7.

        Returns:
            _type_: A ResNet-101 model.
        """
        return ResNet(
            3, 4, 23, 3, num_classes=num_classes, init_kernel_size=init_kernel_size
        )

    @staticmethod
    def make_resnet152(num_classes: int, init_kernel_size: int = 7):
        """Create a ResNet-152 model.

        Args:
            num_classes (int): The number of classes to classify.
            init_kernel_size (int, optional): The number of the classes to classify.
            Defaults to 7.

        Returns:
            _type_: A ResNet-152 model.
        """
        return ResNet(
            3, 8, 36, 3, num_classes=num_classes, init_kernel_size=init_kernel_size
        )
