import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt


def data_generator(dimension, file_name):
    average = np.random.rand(dimension)
    std_dev = np.random.randn(dimension)

    result = np.zeros((50, dimension))
    for i in range(50):
        result[i] = average + (std_dev * np.random.randn(1))

    print(np.max(result))
    print(np.min(result))

    np.save(file_name, result)


def append(arr1, arr2):
    if arr1 is None:
        arr1 = arr2
    else:
        arr1 = np.append(arr1, arr2, axis=0)

    return arr1


if __name__ == "__main__":
    data_generation_phase = False
    NUM_EPOCH = 100
    BATCH_SIZE = 10
    NUM_CLASSES = 5
    NUM_DATA_PER_CLASS = 50

    base_img_file_name = "./image"
    base_txt_file_name = "./text"

    if data_generation_phase == True:
        for i in range(NUM_CLASSES): # image data generation
            file_name = os.path.join(base_img_file_name, str(i) + ".npy")
            data_generator(10, file_name)

        for i in range(NUM_CLASSES): # text data generation
            file_name = os.path.join(base_txt_file_name, str(i) + ".npy")
            data_generator(8, file_name)

    img_list = None
    img_label_list = None

    for i in range(NUM_CLASSES): # image data loader
        file_name = os.path.join(base_img_file_name, str(i) + ".npy")
        loaded_data = np.load(file_name, "r") # load actual image file

        label = np.zeros((NUM_DATA_PER_CLASS, NUM_CLASSES), dtype=float)
        for individual_label in label:
            individual_label[i] = 1

        img_list = append(img_list, loaded_data)
        img_label_list = append(img_label_list, label)

    txt_list = None
    txt_label_list = None

    for i in range(NUM_CLASSES): # text data loader
        file_name = os.path.join(base_txt_file_name, str(i) + ".npy")
        loaded_data = np.load(file_name, "r") # load actual image file

        label = np.zeros((NUM_DATA_PER_CLASS, NUM_CLASSES), dtype=float)
        for individual_label in label:
            individual_label[i] = 1

        txt_list = append(txt_list, loaded_data)
        txt_label_list = append(txt_label_list, label)

    img_list = np.float64(img_list)
    txt_list = np.float64(txt_list)

    img_dataset = tf.data.Dataset.from_tensor_slices((img_list, img_label_list))
    img_dataset = img_dataset.shuffle(buffer_size=NUM_DATA_PER_CLASS * NUM_CLASSES)
    img_dataset = img_dataset.batch(BATCH_SIZE)

    img_test_dataset = tf.data.Dataset.from_tensor_slices((img_list, img_label_list))
    img_test_dataset = img_test_dataset.batch(NUM_DATA_PER_CLASS * NUM_CLASSES)

    iterator = tf.data.Iterator.from_structure(img_dataset.output_types, img_dataset.output_shapes)
    X, Y = iterator.get_next()

    txt_dataset = tf.data.Dataset.from_tensor_slices((txt_list, txt_label_list))
    txt_dataset = txt_dataset.shuffle(buffer_size=NUM_DATA_PER_CLASS * NUM_CLASSES)
    txt_dataset = txt_dataset.batch(BATCH_SIZE)

    txt_test_dataset = tf.data.Dataset.from_tensor_slices((txt_list, txt_label_list))
    txt_test_dataset = txt_test_dataset.batch(NUM_DATA_PER_CLASS * NUM_CLASSES)

    txt_iterator = tf.data.Iterator.from_structure(txt_dataset.output_types, txt_dataset.output_shapes)
    txtX, txtY = txt_iterator.get_next()

    img_init_op = iterator.make_initializer(img_dataset)
    img_test_init_op = iterator.make_initializer(img_test_dataset)

    txt_init_op = txt_iterator.make_initializer(txt_dataset)
    txt_test_inip_op = txt_iterator.make_initializer(txt_test_dataset)

    joint_dataset = tf.data.Dataset.from_tensor_slices((img_list, txt_list, img_label_list))
    joint_dataset = joint_dataset.shuffle(buffer_size=NUM_DATA_PER_CLASS * NUM_CLASSES)
    joint_dataset = joint_dataset.batch(BATCH_SIZE)

    joint_iterator = tf.data.Iterator.from_structure(joint_dataset.output_types, joint_dataset.output_shapes)
    I, T, L = joint_iterator.get_next()

    joint_init_op = joint_iterator.make_initializer(joint_dataset)

    def image_classifier(x, reuse=False):
        with tf.variable_scope("image") as scope:
            if reuse:
                scope.reuse_variables()

            h1 = tf.layers.dense(x, 8)
            h1_activation = tf.nn.relu(h1)

            h2 = tf.layers.dense(h1_activation, 6)
            h2_activation = tf.nn.relu(h2)

            h3 = tf.layers.dense(h2_activation, 4)
            h3_activation = tf.nn.relu(h3)

            h4 = tf.layers.dense(h3_activation, 2)
            logit = tf.layers.dense(h4, NUM_CLASSES)

        return logit, h4


    def text_classifier(x, reuse=False):
        with tf.variable_scope("text") as scope:
            if reuse:
                scope.reuse_variables()

            h1 = tf.layers.dense(x, 6)
            h1_activation = tf.nn.relu(h1)

            h2 = tf.layers.dense(h1_activation, 4)
            h2_activation = tf.nn.relu(h2)

            h3 = tf.layers.dense(h2_activation, 2)
            logit = tf.layers.dense(h3, NUM_CLASSES)

        return logit, h3


    img_classify = image_classifier(X)
    txt_classify = text_classifier(txtX)

    joint_img_feature = image_classifier(I, reuse=True)
    joint_txt_feature = text_classifier(T, reuse=True)

    random_txt0 = tf.placeholder(dtype=tf.float64, shape=(None, 8))
    random_txt1 = tf.placeholder(dtype=tf.float64, shape=(None, 8))
    random_txt2 = tf.placeholder(dtype=tf.float64, shape=(None, 8))
    random_txt3 = tf.placeholder(dtype=tf.float64, shape=(None, 8))
    random_txt4 = tf.placeholder(dtype=tf.float64, shape=(None, 8))

    random_img0 = tf.placeholder(dtype=tf.float64, shape=(None, 10)) # think about use dictionary
    random_img1 = tf.placeholder(dtype=tf.float64, shape=(None, 10))
    random_img2 = tf.placeholder(dtype=tf.float64, shape=(None, 10))
    random_img3 = tf.placeholder(dtype=tf.float64, shape=(None, 10))
    random_img4 = tf.placeholder(dtype=tf.float64, shape=(None, 10))

    rand_txt_feature = list()
    rand_txt_feature.append(text_classifier(random_txt0, reuse=True))
    rand_txt_feature.append(text_classifier(random_txt1, reuse=True))
    rand_txt_feature.append(text_classifier(random_txt2, reuse=True))
    rand_txt_feature.append(text_classifier(random_txt3, reuse=True))
    rand_txt_feature.append(text_classifier(random_txt4, reuse=True))

    rand_img_feature = list()
    rand_img_feature.append(image_classifier(random_img0, reuse=True))
    rand_img_feature.append(image_classifier(random_img1, reuse=True))
    rand_img_feature.append(image_classifier(random_img2, reuse=True))
    rand_img_feature.append(image_classifier(random_img3, reuse=True))
    rand_img_feature.append(image_classifier(random_img4, reuse=True))

    t_vars = tf.global_variables()

    img_vars = [var for var in t_vars if var.name.startswith("image")]
    txt_vars = [var for var in t_vars if var.name.startswith("text")]

    img_loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=img_classify[0],
                                                          labels=Y)
    img_optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(img_loss, var_list=img_vars)

    txt_loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=txt_classify[0],
                                                          labels=txtY)
    txt_optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(txt_loss, var_list=txt_vars)

    init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

    j_img_loss = 0
    j_txt_loss = 0
    for i in range(NUM_CLASSES):
        # rand_idx = np.random.randint(0, NUM_DATA_PER_CLASS - 1)
        j_img_loss += tf.maximum(tf.cast(0.0, tf.float64),
                                 (1 - tf.cast(tf.equal(L, i), tf.float64)) +
                                 tf.reduce_sum(tf.multiply(joint_img_feature[1], rand_txt_feature[i][1]), 1, keep_dims=True) -
                                 tf.reduce_sum(tf.multiply(joint_img_feature[1], joint_txt_feature[1]), 1, keep_dims=True))

        # rand_idx = np.random.randint(0, NUM_DATA_PER_CLASS - 1)
        j_txt_loss += tf.maximum(tf.cast(0.0, tf.float64),
                                 (1 - tf.cast(tf.equal(L, i), tf.float64)) +
                                 tf.reduce_sum(tf.multiply(rand_img_feature[i][1], joint_txt_feature[1]), 1, keep_dims=True) -
                                 tf.reduce_sum(tf.multiply(joint_img_feature[1], joint_txt_feature[1]), 1, keep_dims=True))

    j_total_loss = j_img_loss + j_txt_loss
    j_optimizer = tf.train.AdamOptimizer(learning_rate=0.0005).minimize(j_total_loss, var_list=t_vars)
    j_img_optimizer = tf.train.AdamOptimizer(learning_rate=0.0005).minimize(j_img_loss, var_list=img_vars)
    j_txt_optimizer = tf.train.AdamOptimizer(learning_rate=0.0005).minimize(j_txt_loss, var_list=txt_vars)

    with tf.Session() as sess:
        saver_img = tf.train.Saver(max_to_keep=100, var_list=img_vars)
        saver_txt = tf.train.Saver(max_to_keep=100, var_list=txt_vars)

        sess.run(init)

        # for cur_epoch in range(NUM_EPOCH):
        #     sess.run(img_init_op)
        #
        #     while True:
        #         try:
        #             _, l = sess.run([img_optimizer, img_loss])
        #             print("loss : %f" % np.mean(l))
        #         except tf.errors.OutOfRangeError:
        #             if cur_epoch == NUM_EPOCH - 1:
        #                 saver_img.save(sess, "./pretrained_img.ckpt")
        #             break

        # for cur_epoch in range(NUM_EPOCH):
        #     sess.run(txt_init_op)
        #
        #     while True:
        #         try:
        #             _, l = sess.run([txt_optimizer, txt_loss])
        #             print("loss : %f" % np.mean(l))
        #         except tf.errors.OutOfRangeError:
        #             if cur_epoch == NUM_EPOCH - 1:
        #                 saver_txt.save(sess, "./pretrained_txt.ckpt")
        #             break

        tf.initialize_all_variables().run()
        saver_img.restore(sess, "./pretrained_img.ckpt")
        saver_txt.restore(sess, "./pretrained_txt.ckpt")
        # sess.run(img_test_init_op)
        # _, result = sess.run(img_classify)

        # sess.run(txt_test_inip_op)
        # _, result = sess.run(txt_classify)

        for cur_epoch in range(NUM_EPOCH):
            sess.run(joint_init_op)

            while True:
                try:
                    _, _, l = sess.run([j_img_optimizer, j_txt_optimizer, j_total_loss],
                                    feed_dict={random_txt0: np.tile(np.expand_dims(txt_list[np.random.randint(0, 49)], axis=0), (10, 1)),
                                               random_txt1: np.tile(np.expand_dims(txt_list[np.random.randint(50, 99)], axis=0), (10, 1)),
                                               random_txt2: np.tile(np.expand_dims(txt_list[np.random.randint(100, 149)], axis=0), (10, 1)),
                                               random_txt3: np.tile(np.expand_dims(txt_list[np.random.randint(150, 199)], axis=0), (10, 1)),
                                               random_txt4: np.tile(np.expand_dims(txt_list[np.random.randint(200, 249)], axis=0), (10, 1)),
                                               random_img0: np.tile(np.expand_dims(img_list[np.random.randint(0, 49)], axis=0), (10, 1)),
                                               random_img1: np.tile(np.expand_dims(img_list[np.random.randint(50, 99)], axis=0), (10, 1)),
                                               random_img2: np.tile(np.expand_dims(img_list[np.random.randint(100, 149)], axis=0), (10, 1)),
                                               random_img3: np.tile(np.expand_dims(img_list[np.random.randint(150, 199)], axis=0), (10, 1)),
                                               random_img4: np.tile(np.expand_dims(img_list[np.random.randint(200, 249)], axis=0), (10, 1)),})
                    print("loss: %f" % np.mean(l))
                except tf.errors.OutOfRangeError:
                    break

        sess.run(txt_test_inip_op)
        _, result = sess.run(txt_classify)

        for i in range(NUM_CLASSES):
            for j in range(NUM_DATA_PER_CLASS):
                x, y = result[i * NUM_DATA_PER_CLASS + j]
                plt.scatter(x, y, c='C' + str(i))

        plt.ylim(-22, 12)
        plt.xlim(-33, 28)
        plt.show()

        sess.run(img_test_init_op)
        _, result = sess.run(img_classify)

        for i in range(NUM_CLASSES):
            for j in range(NUM_DATA_PER_CLASS):
                x, y = result[i * NUM_DATA_PER_CLASS + j]
                plt.scatter(x, y, c='C' + str(i))

        plt.ylim(-2, 32)
        plt.xlim(-33, 28)
        plt.show()