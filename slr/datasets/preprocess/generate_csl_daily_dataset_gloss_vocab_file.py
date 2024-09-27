import os.path
import pickle


def generate_csl_daily_dataset_gloss_vocab_file(data_dir: str, output_dir: str):
    data_dir = os.path.abspath(data_dir)
    output_dir = os.path.abspath(output_dir)
    annotation_file = os.path.join(data_dir, "sentence_label/csl2020ct_v2.pkl")
    output_file = os.path.join(output_dir, "csl_daily_gloss_vocab.txt")

    with open(annotation_file, 'rb') as f:
        data = pickle.load(f)

    with open(output_file, 'w', encoding='utf-8') as f:
        for word in data['gloss_map']:
            f.write(f"{word}\n")


if __name__ == '__main__':
    data_dir = "/new_home/xzj23/workspace/SLR/data/csl-daily"
    vocab_dir = "/new_home/xzj23/workspace/SLR/slr/datasets/vocabs"

    generate_csl_daily_dataset_gloss_vocab_file(data_dir=data_dir, output_dir=vocab_dir)
