import tensorflow as tf
import numpy as np


class DS_SJE_model():
    def __init__(self, **args):
        self.args = args['args']
        print(self.args)


    def DS_SJE(self, text_input, reuse=False):
        random_number = np.random.randint(0, 10)
        text_input = text_input[:, random_number]

        with tf.variable_scope("text_encoder") as scope:
            if reuse:
                scope.reuse_variables()

            conv1 = tf.nn.conv2d(input=text_input,
                                 filter=[4, 1, self.args.alphabet_size, 384],
                                 strides=[1, 1, 1, 1],
                                 padding='VALID') # 198 x 1 x 384 dimension
            conv1_activation = tf.nn.relu(conv1)
            conv1_max_pool = tf.nn.max_pool(value=conv1_activation,
                                            ksize=[1, 3, 1, 1],
                                            strides=[1, 3, 1, 1],
                                            padding='VALID') # 66 x 1 x 384 dimension

            conv2 = tf.nn.conv2d(input=conv1_max_pool,
                                 filter=[4, 1, 384, 512],
                                 strides=[1, 1, 1, 1],
                                 padding='VALID') # 63 x 1 x 512 dimension
            conv2_activation = tf.nn.relu(conv2)
            conv2_max_pool = tf.nn.max_pool(value=conv2_activation,
                                            ksize=[1, 3, 1, 1],
                                            strides=[1, 3, 1, 1],
                                            padding='VALID') # 21 x 1 x 512 dimension

            conv3 = tf.nn.conv2d(input=conv2_max_pool,
                                 filter=[5, 1, 512, self.args.cnn_represent_dim],
                                 strides=[1, 1, 1, 1],
                                 padding='VALID') # 18 x 1 x 1024 dimension
            conv3_activation = tf.nn.relu(conv3)
            conv3_max_pool = tf.nn.max_pool(value=conv3_activation,
                                            ksize=[1, 3, 1, 1],
                                            strides=[1, 2, 1, 1],
                                            padding='VALID') # 8 x 1 x 1024 dimension
            cnn_final = tf.squeeze(conv3_max_pool) # 8 x 1024 dimension

            rnn_cell = tf.nn.rnn_cell.BasicRNNCell(self.args.cnn_represent_dim)
            outputs, state = tf.nn.static_rnn(rnn_cell, cnn_final)

            embedded_code = tf.reduce_mean(state, axis=0)

        if reuse == False:
            print('DS_SJE Architecture')
            print(np.shape(conv1_activation))
            print(np.shape(conv1_max_pool))
            print(np.shape(conv2_activation))
            print(np.shape(conv2_max_pool))
            print(np.shape(conv3_activation))
            print(np.shape(conv3_max_pool))
            print(np.shape(embedded_code))

        return embedded_code