import os
import numpy as np
import mindspore as ms
import mindspore.context as context
import resnet, resnetv2, dataset
from mindspore import nn, Tensor
from mindspore.train import Model
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, LossMonitor
from mindspore.train.serialization import load_checkpoint, load_param_into_net


CKPT_1 = 'ckpt/lenet-2_1875.ckpt'

def resume_train(data_dir, ckpt_name="lenet"):
    max_epochs = 1
    num_classes = 10
    lr = 0.00001
    momentum = 0.9
    num_classes = 10
    resnet_shape = [3, 4, 6, 3]
    dataset_sink = context.get_context('device_target') == 'CPU'    ##如果是GPU版本就用GPU
    repeat = max_epochs if dataset_sink else 1
    ds_train = dataset.dataset_train
    ds_test = dataset.dataset_test
    steps_per_epoch = ds_train.get_dataset_size()

    # net = resnetv2.PreActResNet50(num_classes)
    net = resnet.resnet50(num_classes)
    loss = nn.loss.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
    opt = nn.Momentum(net.trainable_params(), lr, momentum)

    param_dict = load_checkpoint(CKPT_1)
    load_param_into_net(net, param_dict)
    load_param_into_net(opt, param_dict)

    ckpt_cfg = CheckpointConfig(save_checkpoint_steps=steps_per_epoch, keep_checkpoint_max=5)
    ckpt_cb = ModelCheckpoint(prefix=ckpt_name, directory='ckpt', config=ckpt_cfg)
    loss_cb = LossMonitor(steps_per_epoch)

    model = Model(net, loss, opt, metrics={'acc', 'loss'})
    model.train(max_epochs, ds_train, callbacks=[ckpt_cb, loss_cb], dataset_sink_mode=dataset_sink)

    metrics = model.eval(ds_test, dataset_sink_mode=dataset_sink)
    print('Metrics:', metrics)

resume_train('/model/ckpt/')
print('Checkpoints after resuming training:')
print('\n'.join(sorted([x for x in os.listdir('ckpt') if x.startswith('train_resnet_unsw_nb15')])))