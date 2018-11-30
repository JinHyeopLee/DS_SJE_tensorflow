import tensorflow as tf
import numpy as np


class DS_SJE_model():
    def __init__(self, **args):
        self.args = args['args']
        print(self.args)


    def DS_SJE(self, text_input, reuse=False):
        text_input = text_input[:, np.random.randint(0, 10)]

        with tf.variable_scope("text_encoder") as scope:
            if reuse:
                scope.reuse_variables()

            conv1 = tf.layers.conv2d(inputs=text_input,
                                     filters=384,
                                     kernel_size=[4, 1],
                                     padding='VALID',
                                     activation=tf.nn.relu) # 198 x 384 dimension
            conv1_max_pool = tf.nn.max_pool(value=conv1,
                                            ksize=[1, 3, 1, 1],
                                            strides=[1, 3, 1, 1],
                                            padding='VALID') # 66 x 1 x 384 dimension

            conv2 = tf.layers.conv2d(inputs=conv1_max_pool,
                                     filters=512,
                                     kernel_size=[4, 1],
                                     padding='VALID',
                                     activation=tf.nn.relu) # 63 x 1 x 512 dimension
            conv2_max_pool = tf.nn.max_pool(value=conv2,
                                            ksize=[1, 3, 1, 1],
                                            strides=[1, 3, 1, 1],
                                            padding='VALID') # 21 x 1 x 512 dimension

            conv3 = tf.layers.conv2d(inputs=conv2_max_pool,
                                     filters=self.args.cnn_represent_dim,
                                     kernel_size=[5, 1],
                                     padding='VALID',
                                     activation=tf.nn.relu) # 18 x 1 x 1024 dimension
            conv3_max_pool = tf.nn.max_pool(value=conv3,
                                            ksize=[1, 3, 1, 1],
                                            strides=[1, 2, 1, 1],
                                            padding='VALID') # 8 x 1 x 1024 dimension
            cnn_final = tf.squeeze(conv3_max_pool, [2]) # 8 x 1024 dimension

            cnn_final_sequence = tf.split(cnn_final, 8, axis=1)
            cnn_final_list = list()
            for temporal in cnn_final_sequence:
                cnn_final_list.append(tf.squeeze(temporal, [1]))

            rnn_cell = tf.nn.rnn_cell.BasicRNNCell(self.args.cnn_represent_dim)
            outputs, state = tf.nn.static_rnn(rnn_cell, cnn_final_list, dtype=tf.float32)

            # embedded_code = tf.reduce_mean(state, axis=1)

        if reuse == False:
            print('DS_SJE Architecture')
            print(np.shape(conv1))
            print(np.shape(conv1_max_pool))
            print(np.shape(conv2))
            print(np.shape(conv2_max_pool))
            print(np.shape(conv3))
            print(np.shape(conv3_max_pool))
            print(np.shape(cnn_final))
            print(np.shape(state))

        return state