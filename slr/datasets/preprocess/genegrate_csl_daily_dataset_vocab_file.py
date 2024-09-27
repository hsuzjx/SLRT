import pickle


def generate_csl_daily_dataset_vocab_file(annotation_file: str, output_file: str, ):
    with open(annotation_file, 'rb') as f:
        data = pickle.load(f)

    with open(output_file, 'w', encoding='utf-8') as f:
        for word in data['gloss_map']:
            f.write(f"{word}\n")


if __name__ == '__main__':
    generate_csl_daily_dataset_vocab_file(
        annotation_file='/new_home/xzj23/workspace/SLR/data/csl-daily/sentence_label/csl2020ct_v2.pkl',
        output_file='/new_home/xzj23/workspace/SLR/slr/datasets/vocabs/csl_daily_gloss_vocab.txt',
    )
