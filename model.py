import tensorflow as tf
import numpy as np


# class CustomRNN(tf.nn.rnn_cell.BasicRNNCell):
#     def __init__(self, num_units, activation=None, reuse=None, dtype=None, **kwargs):
#         # kwargs['state_is_tuple'] = False
#         returns = super(CustomRNN, self).__init__(num_units=num_units,**kwargs)
#         # self._output_size = self._state_size
#         return returns
#
#
#     def __call__(self, inputs, state):
#         output, next_state = super(CustomRNN, self).__call__(inputs, state)
#         return next_state, next_state


class DS_SJE_model():
    def __init__(self, **args):
        self.args = args['args']
        print(self.args)


    def DS_SJE(self, text_input, reuse=False, forward=False):
        if not forward:
            random_number = tf.random.uniform(shape=[1], minval=0, maxval=9, dtype=tf.int32)
            text_input = text_input[:, random_number[0]]

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

            rnn_cell = tf.nn.rnn_cell.BasicRNNCell(self.args.cnn_represent_dim, activation=tf.nn.relu, reuse=reuse)
            # rnn_cell = tf.contrib.cudnn_rnn.CudnnRNNRelu(8, self.args.cnn_represent_dim)
            outputs, state = tf.nn.static_rnn(rnn_cell, cnn_final_list, dtype=tf.float32)

            embedded_code = tf.reduce_mean(outputs, axis=0)

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
            print(np.shape(outputs))
            print(np.shape(embedded_code))

        return embedded_code