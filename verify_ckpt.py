import os
import numpy as np
import mindspore as ms
from mindspore.nn import Accuracy, Loss
import mindspore.context as context
import resnet, resnetv2, dataset
from mindspore import nn, Tensor
from mindspore.train import Model
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, LossMonitor
from mindspore.train.serialization import load_checkpoint, load_param_into_net

def infer():
    lr = 0.00001
    momentum = 0.9
    num_classes = 10
    resnet_shape = [3, 4, 6, 3]
    # net = resnetv2.PreActResNet50(num_classes)
    net = resnet.resnet50(num_classes)
    param_dict = load_checkpoint("./model/ckpt/train_resnet_unsw_nb15_1-37_1852.ckpt")
    load_param_into_net(net, param_dict)

    classes = ["Analysis","Backdoor","DoS","Exploits","Fuzzers",
              "Generic","Normal","Reconnaissance","Shellcode","Worms"]

    data = dataset.dataset_test
    data = data.create_dict_iterator()

    correct = 0
    incorrect = 0
    id = 1

    loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
    opt = nn.Momentum(net.trainable_params(), lr, momentum, weight_decay=0.85)
    metrics = {
        'accuracy': Accuracy(),
        'loss': Loss()
    }
    model = Model(net, loss, opt, metrics=metrics)

    for i in data:
        pred = model.predict(i['data'])
        predicted = int(pred[0].argmax(0))
        actual = int(i['label'])
        if(predicted==actual):
            correct=correct+1
        else:
            incorrect=incorrect+1
        print(f'{id}:Predicted:"{predicted}", Actual:"{actual}"')
        id=id+1

    print(f"Precision is {0}".format(float(correct)/(float(correct)+float(incorrect))))


infer()
