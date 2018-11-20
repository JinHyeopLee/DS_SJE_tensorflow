import tensorflow as tf
from tensorflow.contrib import slim
import tensorflow.contrib.slim.nets
from glob import glob
import os
import cv2
import numpy as np


read_list = list()

with open("/home/jh/CUB/trainvalclasses.txt") as f:
    while True:
        line = f.readline()
        if not line: break
        read_list.append(line)

image_file_name_list = list()

for class_name in read_list:
    class_name = class_name.replace("\n", "")
    new_image_list = sorted(glob(os.path.join("/home/jh/CUB/images",
                                              class_name,
                                              "*.jpg")))
    image_file_name_list = image_file_name_list + new_image_list

images = tf.placeholder(float, (None, 224, 224, 3))

inception = slim.nets.inception

with slim.arg_scope(inception.inception_v1_arg_scope()):
    _, end_points = inception.inception_v1(inputs=images, is_training=False, num_classes=1001)

ckpt_path = "/media/jh/data/pretrained_model/inception_v1.ckpt" # Fix this part!

variables_to_restore = slim.get_variables_to_restore(exclude=[])
saver = tf.train.Saver(variables_to_restore)

with tf.Session() as sess:
    saver.restore(sess, ckpt_path)

    for image_file_name in image_file_name_list:
        image = cv2.imread(image_file_name)

        width = image.shape[1]
        height = image.shape[0]

        print(width, height)

        image_list = list()

        if width > height:
            width = int(width * 256 / height)
            height = 256

            image = cv2.resize(image, (width, height))

            image_list.append(image[0:224, 0:224])  # left-up
            image_list.append(image[0:224, -224:])  # right-up
            image_list.append(image[-224:, 0:224])  # left-down
            image_list.append(image[-224:, -224:])  # right-down
            image_list.append(image[int(height / 2) - 112:int(height / 2) + 112,
                              int(width / 2) - 112:int(width / 2) + 112])  # center

            image = cv2.flip(image, 0)

            image_list.append(image[0:224, 0:224])  # left-up
            image_list.append(image[0:224, -224:])  # right-up
            image_list.append(image[-224:, 0:224])  # left-down
            image_list.append(image[-224:, -224:])  # right-down
            image_list.append(image[int(height / 2) - 112:int(height / 2) + 112,
                              int(width / 2) - 112:int(width / 2) + 112])  # center

        elif height >= width:
            height = int(height * 256 / width)
            width = 256

            image = cv2.resize(image, (width, height))

            image_list.append(image[0:224, 0:224])  # left-up
            image_list.append(image[0:224, -224:])  # right-up
            image_list.append(image[-224:, 0:224])  # left-down
            image_list.append(image[-224:, -224:])  # right-down
            image_list.append(image[int(height / 2) - 112:int(height / 2) + 112,
                              int(width / 2) - 112:int(width / 2) + 112])  # center

            image = cv2.flip(image, 0)

            image_list.append(image[0:224, 0:224])  # left-up
            image_list.append(image[0:224, -224:])  # right-up
            image_list.append(image[-224:, 0:224])  # left-down
            image_list.append(image[-224:, -224:])  # right-down
            image_list.append(image[int(height / 2) - 112:int(height / 2) + 112,
                              int(width / 2) - 112:int(width / 2) + 112])  # center

        represents = np.zeros((10, 1024), dtype=float)

        i = 0
        for image in image_list:
            image = np.expand_dims(image, axis=0)
            end_point = sess.run(end_points, feed_dict={images: image})
            represents[i] = end_point['AvgPool_0a_7x7']
            i += 1

        image_file_name = image_file_name.replace('.jpg', '.npy')
        np.save(image_file_name, represents)