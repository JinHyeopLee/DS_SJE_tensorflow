import numpy as np
from model import DS_SJE_model
import tensorflow as tf
from glob import glob
import os
from DS_SJE_utils import append_nparr


class DS_SJE_evaluation():
    def __init__(self, args):
        self.args = args
        print(self.args)

        self.raw_text = tf.placeholder(dtype=tf.float32,
                                       shape=(None, self.args.length_char_string, 1, self.args.alphabet_size))


    def evaluate(self):
        self.input_pipeline_setup()
        self.classify()
        self.retrieval()


    def input_pipeline_setup(self):
        self.class_list = list()

        self.img_list = None
        self.lbl_list = None

        self.class_txt_list = list()

        if self.args.valid:
            with open(self.args.valid_meta_path, "r") as f:
                while True:
                    line = f.readline()
                    if not line: break
                    self.class_list.append(line)
        else:
            with open(self.args.test_meta_path, "r") as f:
                while True:
                    line = f.readline()
                    if not line: break
                    self.class_list.append(line)

        sess = tf.Session()
        init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

        model = DS_SJE_model(args=self.args)
        forward = model.DS_SJE(self.raw_text, forward=True)
        # class_embed = tf.reduce_mean(forward)

        sess.run(init)
        # tf.initialize_all_variables().run(sess)

        saver = tf.train.Saver()
        saver.restore(sess, self.args.write_model_path + "-299")

        i = 0
        for class_name in self.class_list:
            class_name = class_name.replace("\n", "")
            new_img_file_name_list = sorted(glob(os.path.join(self.args.train_img_path,
                                                              class_name,
                                                              self.args.train_img_data_type)))
            new_img_list = None
            for image_file_name in new_img_file_name_list:  # load actual image file
                new_img = np.load(image_file_name, "r")
                new_img = np.float32(new_img)
                # new_img = np.expand_dims(new_img, axis=0)
                new_img_list = append_nparr(new_img_list, new_img)

            new_txt_file_name_list = sorted(glob(os.path.join(self.args.train_txt_path,
                                                              class_name,
                                                              self.args.train_txt_data_type)))
            new_txt_list = None
            for text_file_name in new_txt_file_name_list:  # load actual text file
                new_txt = np.load(text_file_name, "r")
                new_txt = np.int8(new_txt)
                new_txt_list = append_nparr(new_txt_list, new_txt)

            print(np.shape(new_txt_list))
            result = sess.run(forward, feed_dict={self.raw_text: new_txt_list})
            result = np.mean(result, axis=0)
            self.class_txt_list.append(result)

            new_lbl_list = np.zeros((np.shape(new_img_list)[0], self.args.train_num_classes))
            for j in range(np.shape(new_img_list)[0]):
                new_lbl_list[j, i] = 1

            self.img_list = append_nparr(self.img_list, new_img_list)
            self.lbl_list = append_nparr(self.lbl_list, new_lbl_list)

            i += 1

        self.img_list = np.squeeze(self.img_list)
        print(np.shape(self.img_list))


    def classify(self):
        right_cnt = 0
        overall = np.shape(self.img_list)[0]

        i = 0
        for image in self.img_list:
            sim_list = np.zeros((self.args.valid_num_classes))
            j = 0
            for class_avg in self.class_txt_list:
                sim_list[j] = np.dot(image, class_avg)
                j += 1

            if np.argmax(sim_list) == np.argmax(self.lbl_list[i]):
                right_cnt += 1

            i += 1

        accuracy = right_cnt / overall
        print("Classifiy result: %f" % accuracy)


    def retrieval(self):
        total_precision = 0

        i = 0
        for class_avg in self.class_txt_list:
            sim_list = np.zeros((np.shape(self.img_list)[0]))
            j = 0
            for image in self.img_list:
                sim_list[j] = np.dot(image, class_avg)
                j += 1

            right_cnt = 0
            for k in range(50):
                if np.argmax(self.lbl_list[np.argmax(sim_list)]) == i:
                    right_cnt += 1
                sim_list[np.argmax(sim_list)] = 0
            total_precision += right_cnt / 50.0

        total_precision = total_precision / 50
        print("Retireval result: %f" % total_precision)