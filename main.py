import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # argument for path
    parser.add_argument("--write_model_path", default="/media/jh/data/DS_SJE_model")
    parser.add_argument("--train_img_path", default="/home/jh/CUB/images")
    parser.add_argument("--train_txt_path", default="/home/jh/CUB/text_c10")
    parser.add_argument("--train_meta_path", default="/home/jh/CUB/trainclasses.txt")
    parser.add_argument("--valid_meta_path", default="/home/jh/CUB/valclasses.txt")
    parser.add_argument("--test_meta_path", default="/home/jh/CUB/testclasses.txt")

    # argument for text encoder's hyperparameter
    parser.add_argument("--learning_rate", default=0.0007, type=float)
    parser.add_argument("--alphabet_size", default=71, type=int)
    parser.add_argument("--maximum_text_length", default=201, type=int)
    parser.add_argument("--cnn_represent_dim", default=1024, type=int)

    # argument for data loader
    parser.add_argument("--train_img_data_type", default="*.npy")
    parser.add_argument("--train_txt_data_type", default="*.txt")
    parser.add_argument("--length_char_string", default=201, type=int)
    parser.add_argument("--multi_process_num_thread", default=12, type=int)

    args = parser.parse_args()

    # put more code!