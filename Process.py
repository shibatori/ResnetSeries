import resnet, resnetv2
import dataset
import numpy as np
import os, time
import mindspore
from mindspore import nn, Tensor, ops
from mindspore.nn import Accuracy, ConfusionMatrix, Loss, MAE, F1, Recall, Precision, TrainOneStepWithLossScaleCell
from mindspore.train.model import Model
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, LossMonitor, TimeMonitor
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindspore import context
import mindspore.dataset.engine as de
from mindspore.dataset import vision, transforms

# epoch_size = 1
batch_size = 256
# step_size = 1
num_classes = 10
lr = 0.000001
momentum = 0.9
resnet_shape = [3, 4, 6, 3]

context.set_context(mode=context.GRAPH_MODE, device_target="CPU")  ##如果mindspore版本为GPU，device_target参数为GPU

ds_train = dataset.dataset_train
ds_test = dataset.dataset_test

# # 数据增强
# composed = transforms.Compose(
#     [
#         vision.Rescale(1.0 / 2, 0),
#         vision.Normalize(mean=(0.1250,), std=(0.2500,)),
#         vision.HWC2CHW()
#     ]
# )
#
# ds_train = ds_train.map(composed, 'data')
# ds_test = ds_test.map(composed, 'data')

##打乱数据集
ds_train = ds_train.shuffle(batch_size)
ds_test = ds_test.shuffle(batch_size)

##形成数据集块
ds_train = ds_train.batch(batch_size)
ds_test = ds_test.batch(batch_size)

##重复16遍数据集块
# ds_train = ds_train.repeat(16)
# ds_test = ds_test.repeat(16)


# epochs = 2
# ds_iter = ds_train.create_dict_iterator(output_numpy=True, num_epochs=epochs)
# for _ in range(epochs):
#     for item in ds_iter:
#         print("item: {}".format(item), flush=True)


##使用resnetv2网络训练
# net = resnetv2.PreActResNet50(num_classes)
##使用resnet50网络训练
net = resnet.resnet50(num_classes)

loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True)
opt = nn.Momentum(net.trainable_params(), lr, momentum, weight_decay=0.0001)

metrics = {
    'accuracy': Accuracy(),
    'loss': Loss(),
    'precision': Precision(),
    'recall': Recall(),
    'f1_score': F1()
}
##定义模型
model = Model(net, loss, opt, metrics=metrics)

##训练参数
batch_num = ds_train.get_dataset_size()
max_epochs = 1

model_path = "./model/ckpt"
os.system('rm -f {0}*.ckpt {0}*.meta'.format(model_path))

##定义回调函数
config_ck = CheckpointConfig(save_checkpoint_steps=batch_num, keep_checkpoint_max=1)
ckpoint_cb = ModelCheckpoint(prefix="train_resnet_unsw_nb15", directory=model_path, config=config_ck)
loss_cb = LossMonitor(batch_num)

##训练神经网络
start_time = time.time()
model.fit(max_epochs, ds_train, ds_test, callbacks=[LossMonitor(1), ckpoint_cb, loss_cb], dataset_sink_mode=False)
accuracy = model.eval(ds_test, dataset_sink_mode=False)  # eval
print("result:", accuracy)
cost_time = time.time() - start_time
print("训练总耗时:   %.1f s" % cost_time)


##验证数据集模块
def verify():
    param_dict = load_checkpoint("./model/ckpt/train_resnet50_unsw_nb15_8-1_685.ckpt")
    load_param_into_net(net, param_dict)

    classes = ["Analysis", "Backdoor", "DoS", "Exploits", "Fuzzers",
               "Generic", "Normal", "Reconnaissance", "Shellcode", "Worms"]

    data = dataset.dataset_test
    data = data.create_dict_iterator()

    correct = 0
    incorrect = 0
    id = 1

    for i in data:
        pred = model.predict(i['data'])
        predicted = classes[int(pred[0].argmax(0))]
        actual = classes[int(i['label'])]
        if (predicted == actual):
            correct = correct + 1
        else:
            incorrect = incorrect + 1
        print(f'{id}:Predicted:"{predicted}", Actual:"{actual}"')
        if id % 100 == 0:
            print(f"------------------{id}条数据-------------------")
        id = id + 1

    print("Precision is {0}".format(float(correct) / (float(correct) + float(incorrect))))
