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
from PIL import Image


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


BATCH_NUM = 10
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


def PaddingImage(image):
    '''Can be used for padding/cut/padding&cut'''
    image_size = image.size
    max_edge = 500
    padding_img = Image.new('RGB', (max_edge, max_edge))
    padding_img.paste(image, (0, 0))
    return padding_img


def batch_generator_creator():
    def __reader__():
        # for _ in range(BATCH_NUM * BATCH_SIZE):
        image_shape = [1, 3, 500, 500]
        label_shape = [1, 1]
        label_files = 'train_list.txt'
        label_dic = dict()
        with open(label_files, 'r') as f:
            lines = f.readlines()
            for line in lines:
                d = line.strip().split(' ')
                label_dic[d[0]] = int(d[1])

        directory = '/paddle/nm/train/'
        sub_dirs = os.listdir(directory)
        for sub_dir in sub_dirs:
            sub_dir = os.path.join(directory, sub_dir)
            if os.path.isdir(sub_dir):
                img_dir = os.path.join(directory, sub_dir)
                imgs = os.listdir(img_dir)
                for img in imgs:
                    img = os.path.join(sub_dir, img)
                    image = cv2.imread(img)
                    image = Image.fromarray(
                        cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                    image = PaddingImage(image)
                    if image == -1:
                        continue
                    image = np.array(image).astype('float32').transpose(
                        [2, 0, 1]).reshape(image_shape)
                    label = label_dic[img[11:]]
                    label = np.array(label).reshape(label_shape).astype(
                        'int64')
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
            name='data', shape=[-1, 3, 500, 500], dtype=np.float32)
        label = fluid.data(name='label', shape=[-1, 1], dtype=np.int64)
        model = VGGNET()
        out = fluid.layers.softmax(model.net(data, 1000))
        loss = fluid.layers.cross_entropy(out, soft_label=False, label=label)
        fluid.optimizer.SGD(
            learning_rate=0.01).minimize(fluid.layers.reduce_sum(loss))

        startup_program.random_seed = 1
        exe.run(startup_program)
        loader = fluid.io.DataLoader.from_generator(
            feed_list=[data, label],
            capacity=25,
            use_double_buffer=True,
            iterable=True)
        loader.set_sample_list_generator(
            fluid.io.shuffle(
                fluid.io.batch(
                    batch_generator_creator(), batch_size=5, drop_last=True),
                buf_size=25),
            places=place)

        cnt = 0
        for _ in range(EPOCH_NUM):
            for data in loader():
                loss_data = exe.run(train_program,
                                    feed=data,
                                    fetch_list=[loss])

            fluid.io.save_inference_model(
                './checkpoints_' + str(cnt), ['data'], [out],
                executor=exe,
                main_program=train_program)
            print("This is the " + str(cnt) + "th epoch. ")
            print("The loss is: " + str(loss_data))
            cnt += 1

        print(loss_data)
