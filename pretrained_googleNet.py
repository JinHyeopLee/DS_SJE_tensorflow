import tensorflow as tf
from tensorflow.contrib import slim
import tensorflow.contrib.slim.nets
from glob import glob
import os


#
# dataset = tf.data.Dataset.from_tensor_slices(image_file_name_list)
# dataset = dataset.map(lambda image_file_name:
#                       tuple(tf.py_func(read_train_data,
#                                        [image_file_name],
#                                        [tf.float32])))

images = tf.placeholder(tf.float32, [None, 224, 224, 3])

inception = tf.contrib.slim.nets.inception_v4
with slim.arg_scope(inception.inception_v4_arg_scope()):
    logits, end_points = inception.inception_v4(inputs=images, is_training=False)

ckpt_path = "/media/jh/data/pretrained_model/inception_v4.ckpt" # Fix this part!

variables_to_restore = slim.get_variables_to_restore(exclude=[])

saver = tf.train.Saver(variables_to_restore)