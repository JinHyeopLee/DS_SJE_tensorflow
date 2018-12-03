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
                sess.run(self.train_iterator.initializer)

                loss = 0
                while True:
                    try:
                        for i in range(self.args.train_num_classes):  # sample random image and text of each class
                            sampled_image = self.train_img_list[random_select(i, self.class_instance_num_list)]
                            sampled_text = self.train_txt_list[random_select(i, self.class_instance_num_list)]
                            self.classes_image[i] = sampled_image[np.random.randint(0, 10)]
                            self.classes_text[i] = sampled_text

                        _, loss, summary = sess.run([self.optimizer, self.loss_total, merged],
                                                    feed_dict={self.classes_text_ph: self.classes_text})
                        # print(loss)
                    except tf.errors.OutOfRangeError:
                        print("[EPOCH_{%02d}] last iter loss: %.8f" % (cur_epoch, loss))

                        saver.save(sess, self.args.write_model_path, global_step=cur_epoch)
                        train_writer.add_summary(summary, cur_epoch)
                        break


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
            print(np.shape(self.train_img_list))
            self.train_txt_list = append_nparr(self.train_txt_list, new_txt_list)
            print(np.shape(self.train_txt_list))
            self.train_lbl_list = append_nparr(self.train_lbl_list, new_lbl_list)

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

        dataset = tf.data.Dataset.from_tensor_slices((self.train_img_list,
                                                      self.train_txt_list,
                                                      self.train_lbl_list))
        # dataset = dataset.map()
        # dataset = dataset.map(self.read_train_data,
        #                       num_parallel_calls=self.args.multi_process_num_thread)
        dataset = dataset.shuffle(buffer_size=10000)
        dataset = dataset.prefetch(self.args.batch_size * self.args.prefetch_multiply)
        dataset = dataset.batch(self.args.batch_size)

        self.train_iterator = dataset.make_initializable_iterator()
        self.img_batch, self.txt_batch, self.lbl_batch = self.train_iterator.get_next()


    def network_and_loss_setup(self):
        # Placeholder setup
        self.classes_text_ph = tf.placeholder(tf.float32, (self.args.train_num_classes,
                                                           10, self.args.length_char_string,
                                                           1, self.args.alphabet_size))

        # Network setup
        model = DS_SJE_model(args=self.args)
        encoded_text = model.DS_SJE(tf.cast(self.txt_batch, tf.float32))  # tf cast can be changed by dataset map
        class_encoded_text = list()

        for i in range(self.args.train_num_classes):
            input = tf.expand_dims(self.classes_text_ph[i], axis=0)
            class_encoded_text.append(model.DS_SJE(input, reuse=True))

        # Loss setup
        self.loss_visual = 0
        self.loss_text = 0

        for i in range(self.args.train_num_classes):
            # random_number = np.random.randint(0, 10)
            random_number = tf.random.uniform(shape=[3], minval=0, maxval=9, dtype=tf.int32)

            self.loss_visual += tf.maximum(tf.cast(0.0, tf.float32),
                                           (1 - tf.cast(tf.equal(tf.argmax(self.lbl_batch), i), tf.float32)) +
                                           tf.reduce_sum(tf.multiply(self.img_batch[:, random_number[0]],
                                                                     class_encoded_text[i]),
                                                         axis=1, keepdims=True) -
                                           tf.reduce_sum(tf.multiply(self.img_batch[:, random_number[1]],
                                                                     encoded_text),
                                                         axis=1, keepdims=True))
            self.loss_text += tf.maximum(tf.cast(0.0, tf.float32),
                                         (1 - tf.cast(tf.equal(tf.argmax(self.lbl_batch), i), tf.float32)) +
                                         tf.reduce_sum(tf.multiply(self.classes_image[i], encoded_text),
                                                       axis=1, keepdims=True) -
                                         tf.reduce_sum(tf.multiply(self.img_batch[:, random_number[2]], encoded_text),
                                                       axis=1, keepdims=True))

        self.loss_total = tf.reduce_sum(self.loss_visual + self.loss_text) / self.args.train_num_classes
        self.optimizer = tf.train.RMSPropOptimizer(learning_rate=self.args.learning_rate).minimize(self.loss_total)

        tf.summary.scalar('total_loss', self.loss_total)