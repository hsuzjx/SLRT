import os.path

import pandas as pd


def generate_phoenix2014T_dataset_gloss_vocab_file(data_dir: str, output_dir: str):
    datasets = []
    for mode in ['train', 'test', 'dev']:
        dataset = pd.read_csv(
            os.path.join(data_dir, "PHOENIX-2014-T/annotations/manual", f"PHOENIX-2014-T.{mode}.corpus.csv"),
            sep='|', header=0, index_col="name")
        datasets.append(dataset)

    # get sentences
    sentences = [line[1]["orth"] for dataset in datasets for line in dataset.iterrows()]

    # get sorted glosses
    glosses = sorted({gloss for sentence in sentences for gloss in sentence.split(' ') if gloss})

    with open(os.path.join(output_dir, "phoenix2014T_gloss_vocab.txt"), "w") as f:
        for gloss in glosses:
            f.write(f"{gloss}\n")


if __name__ == '__main__':
    data_dir = "/new_home/xzj23/workspace/SLR/data/phoenix2014T"
    vocab_dir = "/new_home/xzj23/workspace/SLR/slr/datasets/vocabs"
    generate_phoenix2014T_dataset_gloss_vocab_file(data_dir=data_dir, output_dir=vocab_dir)
