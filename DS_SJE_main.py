import tensorflow as tf
import numpy as np
from glob import glob
import os
from model import DS_SJE_model
from DS_SJE_utils import random_select
from DS_SJE_utils import append_nparr


class DS_SJE():
    def __init__(self, args):
        self.args = args
        print(self.args)

        self.classes_image = np.zeros((self.args.train_num_classes, self.args.cnn_represent_dim), dtype=np.float32)
        self.classes_text = np.zeros((self.args.train_num_classes, 10,
                                      self.args.length_char_string, 1, self.args.alphabet_size), dtype=np.float32)


    def train(self):
        self.input_pipeline_setup()
        self.network_and_loss_setup()

        init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(self.args.write_summary_path)

        with tf.Session() as sess:
            saver = tf.train.Saver(max_to_keep=100)

            sess.run(init)

            for cur_epoch in range(self.args.num_epoch):
                total_loss = 0
                for cur_iter in range(self.args.num_iter_per_epoch):
                    cur_img_batch, cur_txt_batch = self.dataset_get_next()
                    _, loss, summary = sess.run([self.optimizer, self.loss_total, merged],
                                                feed_dict={self.batch_img_ph: cur_img_batch,
                                                           self.batch_txt_ph: cur_txt_batch})
                    total_loss += loss
                    # print(loss)

                print("[EPOCH_{%02d}] total loss: %.8f" % (cur_epoch, total_loss))

                saver.save(sess, self.args.write_model_path, global_step=cur_epoch)
                train_writer.add_summary(summary, cur_epoch)
                if cur_epoch >= self.args.learning_rate_decay_after:
                    self.args.learning_rate = self.args.learning_rate * self.args.learning_rate_decay
                    print(self.args.learning_rate)


    def dataset_get_next(self):
        minibatch_class = np.random.permutation(100)
        minibatch_class = minibatch_class[:self.args.batch_size]

        img_batch = None
        txt_batch = None

        for class_idx in minibatch_class:
            random_idx = random_select(class_idx, self.class_instance_num_list)
            image = self.train_img_list[random_idx]
            image = np.expand_dims(image, axis=0)
            img_batch = append_nparr(img_batch, image)
            text = np.float32(self.train_txt_list[random_idx])
            text = np.expand_dims(text, axis=0)
            txt_batch = append_nparr(txt_batch, text)

        return img_batch, txt_batch


    def input_pipeline_setup(self):
        self.train_class_list = list()
        # valid_list = list()

        with open(self.args.train_meta_path, "r") as f:
            while True:
                line = f.readline()
                if not line: break
                self.train_class_list.append(line)

        self.train_img_list = None
        self.train_txt_list = None
        self.train_lbl_list = None

        self.class_instance_num_list = list()

        i = 0
        for class_name in self.train_class_list:
            class_name = class_name.replace("\n", "")
            new_img_file_name_list = sorted(glob(os.path.join(self.args.train_img_path,
                                                              class_name,
                                                              self.args.train_img_data_type)))
            new_img_list = None
            for image_file_name in new_img_file_name_list:  # load actual image file
                new_img = np.load(image_file_name, "r")
                new_img = np.float32(new_img)
                new_img = np.mean(new_img, axis=0) # add this line for average 10 views of one image
                new_img = np.expand_dims(new_img, axis=0)
                new_img_list = append_nparr(new_img_list, new_img)

            new_txt_file_name_list = sorted(glob(os.path.join(self.args.train_txt_path,
                                                              class_name,
                                                              self.args.train_txt_data_type)))
            new_txt_list = None
            for text_file_name in new_txt_file_name_list:  # load actual text file
                new_txt = np.load(text_file_name, "r")
                new_txt = np.int8(new_txt)
                # new_txt = np.squeeze(new_txt)
                new_txt = np.expand_dims(new_txt, axis=0)
                new_txt_list = append_nparr(new_txt_list, new_txt)

            print(np.shape(new_img_list))

            new_lbl_list = np.zeros((np.shape(new_img_list)[0], self.args.train_num_classes))
            for j in range(np.shape(new_img_list)[0]):
                new_lbl_list[j, i] = 1

            self.class_instance_num_list.append(np.shape(new_img_list)[0])

            self.train_img_list = append_nparr(self.train_img_list, new_img_list)
            self.train_txt_list = append_nparr(self.train_txt_list, new_txt_list)
            self.train_lbl_list = append_nparr(self.train_lbl_list, new_lbl_list)

            print(np.shape(self.train_img_list))
            print(np.shape(self.train_txt_list))

            i += 1

        # with open(self.args.valid_meta_path, "r") as f:
        #     while True:
        #         line = f.readline()
        #         if not line: break
        #         valid_list.append(line)
        #
        # valid_img_file_name_list = list()
        # valid_txt_file_name_list = list()
        # valid_lbl_list = list()
        #
        # i = 0
        # for class_name in valid_list:
        #     class_name = class_name.replace("\n", "")
        #     new_img_list = sorted(glob(os.path.join(self.args.train_img_path,
        #                                             class_name,
        #                                             self.args.train_img_data_type)))
        #     new_txt_list = sorted(glob(os.path.join(self.args.train_txt_path,
        #                                             class_name,
        #                                             self.args.train_txt_data_type)))
        #     new_lbl_list = list()
        #     for _ in range(len(new_img_list)):
        #         class_label = np.zeros(self.args.valid_num_classes, dtype=int)
        #         class_label[i] = 1
        #         new_lbl_list.append(class_label)
        #
        #     valid_img_file_name_list += new_img_list
        #     valid_txt_file_name_list += new_txt_list
        #     valid_lbl_list += new_lbl_list
        #
        #     i += 1
        # dataset = tf.data.Dataset.from_tensor_slices((self.train_img_list,
        #                                               self.train_txt_list,
        #                                               self.train_lbl_list))
        # # dataset = dataset.map()
        # # dataset = dataset.map(self.read_train_data,
        # #                       num_parallel_calls=self.args.multi_process_num_thread)
        # dataset = dataset.shuffle(buffer_size=10000)
        # dataset = dataset.prefetch(self.args.batch_size * self.args.prefetch_multiply)
        # dataset = dataset.batch(self.args.batch_size)
        #
        # self.train_iterator = dataset.make_initializable_iterator()
        # self.img_batch, self.txt_batch, self.lbl_batch = self.train_iterator.get_next()


    def network_and_loss_setup(self):
        # Placeholder setup
        self.batch_txt_ph = tf.placeholder(tf.float32, (None, 10, self.args.length_char_string,
                                                        1, self.args.alphabet_size))
        self.batch_img_ph = tf.placeholder(tf.float32, (None, self.args.cnn_represent_dim))

        # Network setup
        model = DS_SJE_model(args=self.args)
        encoded_text = model.DS_SJE(self.batch_txt_ph)  # tf cast can be changed by dataset map
        # class_encoded_text = list()
        #
        # for i in range(self.args.train_num_classes):
        #     input = tf.expand_dims(self.classes_text_ph[i], axis=0)
        #     class_encoded_text.append(model.DS_SJE(input, reuse=True))

        # Loss setup
        self.loss_visual = 0
        self.loss_text = 0

        for i in range(self.args.batch_size):
            inner_visual_loss = 0
            inner_text_loss = 0

            for j in range(self.args.batch_size):
                inner_visual_loss += 1 - tf.cast(tf.equal(i, j), tf.float32) # check this part
                inner_visual_loss += tf.reduce_sum(tf.multiply(self.batch_img_ph[i], encoded_text[j]))
                inner_visual_loss -= tf.reduce_sum(tf.multiply(self.batch_img_ph[i], encoded_text[i]))

                inner_text_loss += 1 - tf.cast(tf.equal(i, j), tf.float32)
                inner_text_loss += tf.reduce_sum(tf.multiply(self.batch_img_ph[j], encoded_text[i]))
                inner_text_loss -= tf.reduce_sum(tf.multiply(self.batch_img_ph[i], encoded_text[i]))

            inner_visual_loss /= self.args.batch_size
            inner_text_loss /= self.args.batch_size

            self.loss_visual += tf.maximum(tf.cast(0.0, tf.float32), inner_visual_loss)
            self.loss_text += tf.maximum(tf.cast(0.0, tf.float32), inner_text_loss)

        # for i in range(self.args.train_num_classes):
        #     # random_number = np.random.randint(0, 10)
        #     # random_number = tf.random.uniform(shape=[3], minval=0, maxval=9, dtype=tf.int32)
        #
        #     self.loss_visual += tf.maximum(tf.cast(0.0, tf.float32),
        #                                    (1 - tf.cast(tf.equal(tf.argmax(self.lbl_batch), i), tf.float32)) +
        #                                    tf.reduce_sum(tf.multiply(self.img_batch,
        #                                                              class_encoded_text[i]),
        #                                                  axis=1, keepdims=True) -
        #                                    tf.reduce_sum(tf.multiply(self.img_batch,
        #                                                              encoded_text),
        #                                                  axis=1, keepdims=True))
        #     self.loss_text += tf.maximum(tf.cast(0.0, tf.float32),
        #                                  (1 - tf.cast(tf.equal(tf.argmax(self.lbl_batch), i), tf.float32)) +
        #                                  tf.reduce_sum(tf.multiply(self.classes_image[i], encoded_text),
        #                                                axis=1, keepdims=True) -
        #                                  tf.reduce_sum(tf.multiply(self.img_batch, encoded_text),
        #                                                axis=1, keepdims=True))

        self.loss_total = (self.loss_visual + self.loss_text) / self.args.batch_size
        self.optimizer = tf.train.RMSPropOptimizer(learning_rate=self.args.learning_rate).minimize(self.loss_total)

        tf.summary.scalar('total_loss', self.loss_total)