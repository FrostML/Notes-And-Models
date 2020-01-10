from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import paddle.fluid as fluid
import numpy as np
from operator import mul
import functools
from functools import reduce
import os
import cv2


class VGGNET():
    def __init__(self):
        pass

    def net(self, input, num_class):
        layers_num = [2, 2, 3, 3, 3]
        out_layer = input
        num_filters = [64, 128, 256, 512, 512]
        for i in range(5):
            name_layer = "conv" + str(i + 1)
            out_layer = self.conv_layers(
                out_layer,
                layers_num[i],
                num_filter=num_filters[i],
                name=name_layer)

        num_fc_size = [4096, 4096, num_class]
        fc_out = out_layer
        for x in range(3):
            fc_out = fluid.layers.fc(input=fc_out,
                                     size=num_fc_size[x],
                                     act='relu')

        return fc_out

    def conv_layers(self, input, layer_num, num_filter, name):
        conv = input
        for i in range(layer_num):
            conv = fluid.layers.conv2d(
                conv,
                num_filter,
                filter_size=3,
                padding=1,
                stride=1,
                act='relu',
                param_attr=fluid.ParamAttr(name=name + "_weight" + str(i)),
                bias_attr=fluid.ParamAttr(name=name + "_bias" + str(i)))

        return fluid.layers.pool2d(
            conv, pool_size=2, pool_type='max', pool_stride=2)


BATCH_NUM = 100
BATCH_SIZE = 2
EPOCH_NUM = 10


def generate_images_and_labels(image_shape, label_shape):
    image = np.random.randint(
        low=0, high=255, size=image_shape).astype('float32')
    threshold = 0.5
    labels = []
    num = reduce(mul, label_shape)
    for i in range(num):
        if np.random.random(size=None) < threshold:
            labels.append(1)
        else:
            labels.append(0)

    label = np.array(labels).reshape(label_shape).astype('int64')
    return image, label


'''
def get_images_and_labels():
    directory = '/paddle/nm/train/'
    sub_dirs = os.listdir(directory)
    for sub_dir in sub_dirs:
        if os.path.isdir(sub_dir):
            img_dir = directory + sub_dir
            imgs = os.listdir(img_dir)
            for img in imgs:
'''


def batch_generator_creator():
    def __reader__():
        for _ in range(BATCH_NUM * BATCH_SIZE):
            image, label = generate_images_and_labels([2, 3, 224, 224], [2, 1])
            yield image, label

    return __reader__


if __name__ == "__main__":
    use_cuda = True
    place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
    exe = fluid.Executor(place)

    train_program = fluid.Program()
    startup_program = fluid.Program()

    with fluid.program_guard(train_program, startup_program):
        data = fluid.data(
            name='data', shape=[-1, 3, 224, 224], dtype=np.float32)
        label = fluid.data(name='label', shape=[-1, 1], dtype=np.int64)
        model = VGGNET()
        out = fluid.layers.softmax(model.net(data, 100))
        loss = fluid.layers.cross_entropy(out, soft_label=False, label=label)
        fluid.optimizer.SGD(
            learning_rate=0.01).minimize(fluid.layers.reduce_sum(loss))

    startup_program.random_seed = 1
    exe.run(startup_program)
    loader = fluid.io.DataLoader.from_generator(
        feed_list=[data, label],
        capacity=6,
        use_double_buffer=True,
        iterable=True)
    loader.set_batch_generator(batch_generator_creator(), places=place)

    cnt = 0
    for _ in range(EPOCH_NUM):
        for data in loader():
            loss_data = exe.run(train_program, feed=data, fetch_list=[loss])
        print("This is the " + str(cnt) + "th epoch. ")
        print("The loss is: " + str(loss_data))
        cnt += 1

    print(loss_data)
