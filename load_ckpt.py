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


CKPT_1 = 'model/ckpt/train_resnet_unsw_nb15_1-37_1852.ckpt'

def resume_train():
    max_epochs = 1
    num_classes = 10
    batch_size = 1024
    lr = 0.01
    momentum = 0.9
    num_classes = 10
    resnet_shape = [3, 4, 6, 3]

    model_path = "./model/ckpt"
    # os.system('rm -f {0}*.ckpt {0}*.meta'.format(model_path))

    dataset_sink = context.get_context('device_target') == 'CPU'    ##如果是GPU版本就用GPU
    ds_train = dataset.dataset_train
    ds_test = dataset.dataset_test

    ds_train = ds_train.shuffle(batch_size)
    ds_test = ds_test.shuffle(batch_size)

    ds_train = ds_train.batch(batch_size)
    ds_test = ds_test.batch(batch_size)

    # ds_train = ds_train.repeat(2)
    # ds_test = ds_test.repeat(2)

    batch_num = ds_train.get_dataset_size()

    # net = resnetv2.PreActResNet50(num_classes)
    net = resnet.resnet50(num_classes)
    loss = nn.loss.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
    opt = nn.Momentum(net.trainable_params(), lr, momentum, weight_decay=0.9)

    param_dict = load_checkpoint(CKPT_1)
    load_param_into_net(net, param_dict)
    load_param_into_net(opt, param_dict)

    ckpt_cfg = CheckpointConfig(save_checkpoint_steps=batch_num, keep_checkpoint_max=1)
    ckpt_cb = ModelCheckpoint(prefix="train_resnet50_unsw_nb15", directory=model_path, config=ckpt_cfg)
    loss_cb = LossMonitor(batch_num)

    metrics = {
        'accuracy': Accuracy(),
        'loss': Loss()
    }

    model = Model(net, loss, opt, metrics=metrics)
    model.fit(max_epochs, ds_train, ds_test, callbacks=[LossMonitor(1), ckpt_cb, loss_cb], dataset_sink_mode=dataset_sink)

    metrics = model.eval(ds_test, dataset_sink_mode=dataset_sink)
    print('Metrics:', metrics)

resume_train()
print('Checkpoints after resuming training:')
print('\n'.join(sorted([x for x in os.listdir('./model/ckpt') if x.startswith('train_resnet_unsw_nb15')])))