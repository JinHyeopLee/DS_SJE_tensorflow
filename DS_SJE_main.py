import tensorflow as tf
import numpy as np
from glob import glob
import os
from model import DS_SJE_model


class DS_SJE():
    def __init__(self, args):
        self.args = args
        print(self.args)

        self.text_input_format = tf.placeholder(tf.float32,
                                                [None, self.args.maximum_text_length, 1, self.args.alphabet_size])
        self.image_input_format = tf.placeholder(tf.float32,
                                                 [None, self.args.cnn_represent_dim])
        self.label_input_format = tf.placeholder(tf.float32,
                                                 [None, self.args.train_num_classes])

        # self.image_represent_avg = np.zeros((100, 1024))


    def train(self):
        return 0


    # def read_train_data(self, tuple):
    #     image = np.load(tuple[0], "r")
    #     text = np.load(tuple[1], "r")
    #
    #     return image, text, tuple[2]


    def input_pipeline_setup(self):
        train_list = list()
        # valid_list = list()

        with open(self.args.train_meta_path, "r") as f:
            while True:
                line = f.readline()
                if not line: break
                train_list.append(line)

        self.train_img_list = np.array(list())
        self.train_txt_list = np.array(list())
        self.train_lbl_list = np.array(list())

        i = 0
        for class_name in train_list:
            class_name = class_name.replace("\n", "")
            new_img_file_name_list = sorted(glob(os.path.join(self.args.train_img_path,
                                                              class_name,
                                                              self.args.train_img_data_type)))
            new_img_list = np.array(list())
            for image_file_name in new_img_file_name_list: # load actual image file
                new_img = np.load(image_file_name, "r")
                np.expand_dims(new_img, axis=0)
                new_img_list = np.append(new_img_list, new_img, axis=0)

            # self.image_represent_avg[i] = np.mean(new_img_list, axis=0) # class average represenatation

            new_txt_file_name_list = sorted(glob(os.path.join(self.args.train_txt_path,
                                                              class_name,
                                                              self.args.train_txt_data_type)))
            new_txt_list = np.array(list())
            for text_file_name in new_txt_file_name_list: # load actual text file
                new_txt = np.load(text_file_name, "r")
                np.expand_dims(new_txt, axis=0)
                new_txt_list = np.append(new_txt_list, new_txt, axis=0)

            new_lbl_list = np.zeros((np.shape(new_img_list)[0], 100))
            for j in range(np.shape(new_img_list)[0]):
                new_lbl_list[j, i] = 1

            self.train_img_list = np.append(self.train_img_list, new_img_list, axis=0)
            self.train_txt_list = np.append(self.train_txt_list, new_txt_list, axis=0)
            self.train_lbl_list = np.append(self.train_lbl_list, new_lbl_list, axis=0)

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
        # dataset = dataset.map(self.read_train_data,
        #                       num_parallel_calls=self.args.multi_process_num_thread)
        dataset = dataset.shuffle(buffer_size=10000)
        dataset = dataset.prefetch(self.args.batch_size * self.args.prefetch_multiply)
        dataset = dataset.batch(self.args.batch_size)

        self.train_iterator = dataset.make_initializable_iterator()

        return self.train_iterator.get_next()


    def network_and_loss_setup(self):
        # Network setup
        model = DS_SJE_model(args=self.args)
        encoded_text = model.DS_SJE(text_input=self.text_input_format) # to make this part with no feed_dict

        # Loss setup
        self.loss_visual = 0
        self.loss_text = 0

        for i in range(self.args.train_num_classes):
            j = 0 #


        self.loss_total = (self.loss_visual + self.loss_text) / self.args.train_num_classes
