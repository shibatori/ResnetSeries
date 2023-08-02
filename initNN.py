import mindspore
import numpy as np
from mindspore import nn, ops
from mindspore.nn import Conv2d, BatchNorm2d, ReLU, Dense, MaxPool2d, Cell, Flatten
from mindspore.ops.operations import TensorAdd
from mindspore.common.tensor import Tensor
#from mindspore.train.model import Model
#from mindspore.nn import SoftmaxCrossEntropyWithLogits
#from mindspore.nn import Momentum


# 定义变量初始化
def weight_variable(shape):
    ones = np.ones(shape).astype(np.float32)
    return Tensor(ones * 0.01)


# 定义conv
def conv1x1(in_channels, out_channels, stride=1, padding=0):
    weight_shape = (out_channels, in_channels, 1, 1)
    weight = weight_variable(weight_shape)
    return Conv2d(in_channels,
                  out_channels,
                  kernel_size=1,
                  stride=stride,
                  padding=padding,
                  weight_init=weight,
                  has_bias=False,
                  pad_mode="same")


def conv3x3(in_channels, out_channels, stride=1, padding=1):
    weight_shape = (out_channels, in_channels, 3, 3)
    weight = weight_variable(weight_shape)
    return Conv2d(in_channels,
                  out_channels,
                  kernel_size=3,
                  stride=stride,
                  padding=padding,
                  weight_init=weight,
                  has_bias=False,
                  pad_mode="same")


def conv7x7(in_channels, out_channels, stride=1, padding=0):
    weight_shape = (out_channels, in_channels, 7, 7)
    weight = weight_variable(weight_shape)
    return Conv2d(in_channels,
                  out_channels,
                  kernel_size=7,
                  stride=stride,
                  padding=padding,
                  weight_init=weight,
                  has_bias=False,
                  pad_mode="same")


# 定义BatchNorm
def bn_with_initialize(out_channels):
    shape = out_channels
    mean = weight_variable(shape)
    var = weight_variable(shape)
    beta = weight_variable(shape)
    gamma = weight_variable(shape)
    bn = BatchNorm2d(out_channels,
                     momentum=0.1,
                     eps=1e-5,
                     gamma_init=gamma,
                     beta_init=beta,
                     moving_mean_init=mean,
                     moving_var_init=var)
    return bn


# 定义dense
def fc_with_initialize(input_channels, out_channels):
    weight_shape = (out_channels, input_channels)
    bias_shape = out_channels
    weigh = weight_variable(weight_shape)
    bias = weight_variable(bias_shape)
    return Dense(input_channels, out_channels, weigh, bias)


# 定义ResidualBlock
class ResidualBlock(Cell):
    expansion = 4

    def __init__(self,
                 in_channels,
                 out_channels,
                 stride=1,
                 down_sample=False):
        super(ResidualBlock, self).__init__()

        out_chls = out_channels // self.expansion
        self.conv1 = conv1x1(in_channels, out_chls, stride=stride, padding=0)
        self.bn1 = bn_with_initialize(out_chls)

        self.conv2 = conv3x3(out_chls, out_chls, stride=1, padding=0)
        self.bn2 = bn_with_initialize(out_chls)

        self.conv3 = conv1x1(out_chls, out_channels, stride=1, padding=0)
        self.bn3 = bn_with_initialize(out_channels)

        self.relu = ReLU()
        self.add = TensorAdd()

    def construct(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        out = self.add(out, identity)
        out = self.relu(out)
        return out


class ResidualBlockWithDown(Cell):
    expansion = 4

    def __init__(self,
                 in_channels,
                 out_channels,
                 stride=1,
                 down_sample=False):
        super(ResidualBlockWithDown, self).__init__()

        out_chls = out_channels // self.expansion
        self.conv1 = conv1x1(in_channels, out_chls, stride=stride, padding=0)
        self.bn1 = bn_with_initialize(out_chls)

        self.conv2 = conv3x3(out_chls, out_chls, stride=1, padding=0)
        self.bn2 = bn_with_initialize(out_chls)

        self.conv3 = conv1x1(out_chls, out_channels, stride=1, padding=0)
        self.bn3 = bn_with_initialize(out_channels)

        self.relu = ReLU()
        self.downSample = down_sample

        self.conv_down_sample = conv1x1(in_channels, out_channels, stride=stride, padding=0)
        self.bn_down_sample = bn_with_initialize(out_channels)
        self.add = TensorAdd()

    def construct(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        identity = self.conv_down_sample(identity)
        identity = self.bn_down_sample(identity)
        out = self.add(out, identity)
        out = self.relu(out)
        return out


# 定义MakeLayer
class MakeLayer0(Cell):
    def __init__(self, block, layer_num, in_channels, out_channels, stride):
        super(MakeLayer0, self).__init__()
        self.a = ResidualBlockWithDown(in_channels, out_channels, stride, down_sample=True)
        self.b = block(out_channels, out_channels, stride=1)
        self.c = block(out_channels, out_channels, stride=1)

    def construct(self, x):
        x = self.a(x)
        x = self.b(x)
        x = self.c(x)
        return x


class MakeLayer1(Cell):
    def __init__(self, block, layer_num, in_channels, out_channels, stride):
        super(MakeLayer1, self).__init__()
        self.a = ResidualBlockWithDown(in_channels, out_channels, stride, down_sample=True)
        self.b = block(out_channels, out_channels, stride=1)
        self.c = block(out_channels, out_channels, stride=1)
        self.d = block(out_channels, out_channels, stride=1)

    def construct(self, x):
        x = self.a(x)
        x = self.b(x)
        x = self.c(x)
        x = self.d(x)
        return x


class MakeLayer2(Cell):
    def __init__(self, block, layer_num, in_channels, out_channels, stride):
        super(MakeLayer2, self).__init__()
        self.a = ResidualBlockWithDown(in_channels, out_channels, stride, down_sample=True)
        self.b = block(out_channels, out_channels, stride=1)
        self.c = block(out_channels, out_channels, stride=1)
        self.d = block(out_channels, out_channels, stride=1)
        self.e = block(out_channels, out_channels, stride=1)
        self.f = block(out_channels, out_channels, stride=1)

    def construct(self, x):
        x = self.a(x)
        x = self.b(x)
        x = self.c(x)
        x = self.d(x)
        x = self.e(x)
        x = self.f(x)
        return x


class MakeLayer3(nn.Cell):
    def __init__(self, block, layer_num, in_channels, out_channels, stride):
        super(MakeLayer3, self).__init__()
        self.a = ResidualBlockWithDown(in_channels, out_channels, stride, down_sample=True)
        self.b = block(out_channels, out_channels, stride=1)
        self.c = block(out_channels, out_channels, stride=1)

    def construct(self, x):
        x = self.a(x)
        x = self.b(x)
        x = self.c(x)
        return x

class ResNet(Cell):
    def __init__(self, block, layer_num, num_classes=10):
        super(ResNet, self).__init__()
        self.conv1 = conv7x7(4, 64, stride=2, padding=0)

        self.bn1 = bn_with_initialize(64)
        self.relu = ReLU()
        self.maxpool = MaxPool2d(kernel_size=3, stride=2, pad_mode="same")

        self.layer1 = MakeLayer0(
            block, layer_num[0], in_channels=64, out_channels=256, stride=1)
        self.layer2 = MakeLayer1(
            block, layer_num[1], in_channels=256, out_channels=512, stride=2)
        self.layer3 = MakeLayer2(
            block, layer_num[2], in_channels=512, out_channels=1024, stride=2)
        self.layer4 = MakeLayer3(
            block, layer_num[3], in_channels=1024, out_channels=2048, stride=2)

        self.mean = ops.ReduceSum(keep_dims=True)
        self.fc = fc_with_initialize(512*block.expansion, num_classes)
        self.flatten = Flatten()

    def construct(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.mean(x, (2, 3))
        x = self.flatten(x)
        x = self.fc(x)
        return x

    def resnet50(num_classes, resnet_shape):
        return ResNet(ResidualBlock, resnet_shape, num_classes)

    def resnet18(num_classes, resnet_shape):
        return ResNet(ResidualBlock, resnet_shape, num_classes)

'''
dataset_org = np.load('UNSWNB_labelall_test_dataset.npz')
data = dataset_org['data']
net = ResNet()
x = Tensor(np.ones((32, 224, 112, 3, 5), dtype=np.float32), mindspore.float32)
out = net(x)
print('out',out.shape)
'''