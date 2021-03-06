import argparse
from DS_SJE_main import DS_SJE
from evaluation import DS_SJE_evaluation

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # argument for path
    parser.add_argument("--write_model_path", default="/result/DS_SJE_model")
    parser.add_argument("--train_img_path", default="/data/images")
    parser.add_argument("--train_txt_path", default="/data/text_c10")
    parser.add_argument("--train_meta_path", default="/data/trainclasses.txt")
    parser.add_argument("--valid_meta_path", default="/data/valclasses.txt")
    parser.add_argument("--test_meta_path", default="/data/testclasses.txt")

    # argument for text encoder's hyperparameter
    parser.add_argument("--learning_rate", default=0.0007, type=float)
    parser.add_argument("--learning_rate_decay", default=0.98)
    parser.add_argument("--learning_rate_decay_after", default=1)
    parser.add_argument("--alphabet_size", default=70, type=int)
    parser.add_argument("--maximum_text_length", default=201, type=int)
    parser.add_argument("--cnn_represent_dim", default=1024, type=int)
    parser.add_argument("--batch_size", default=40, type=int)
    parser.add_argument("--prefetch_multiply", default=3, type=int)

    # argument for learning
    parser.add_argument("--num_epoch", default=200, type=int)
    parser.add_argument("--num_iter_per_epoch", default=100, type=int)
    parser.add_argument("--write_summary_path", default="./summary")

    # argument for data loader
    parser.add_argument("--train_img_data_type", default="*.npy")
    parser.add_argument("--train_txt_data_type", default="*.npy")
    parser.add_argument("--length_char_string", default=201, type=int)
    parser.add_argument("--multi_process_num_thread", default=12, type=int)
    parser.add_argument("--train_num_classes", default=100, type=int)
    parser.add_argument("--valid_num_classes", default=50, type=int)

    # argument for evaluator
    parser.add_argument("--valid", default=True)
    parser.add_argument("--train", default=True)

    args = parser.parse_args()

    if args.train:
        model = DS_SJE(args=args)
        model.train()
    else:
        model = DS_SJE_evaluation(args=args)
        model.evaluate()