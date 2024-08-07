import os
import numpy as np
import pandas as pd

def preprocess(dataset_name, annotations_path, gloss_dict_path, ground_truth_path):
    # gloss dict
    gloss_dict_file = os.path.join(os.path.abspath(gloss_dict_path), f'{dataset_name}_gloss_dict.npy')
    if not os.path.exists(os.path.dirname(gloss_dict_file)):
        os.makedirs(os.path.dirname(gloss_dict_file))
    if not generate_gloss_dict(gloss_dict_file, annotations_path):
        exit(1)

    # ground truth
    ground_truth_path = os.path.abspath(ground_truth_path)
    for mode in ['train', 'dev', 'test']:
        ground_truth_file = os.path.join(ground_truth_path, f'{dataset_name}-groundtruth-{mode}.stm')
        if not os.path.exists(os.path.dirname(ground_truth_file)):
            os.makedirs(os.path.dirname(ground_truth_file))
        if not generate_ground_truth(ground_truth_file, annotations_path, mode):
            exit(1)


def generate_gloss_dict(gloss_dict_file, annotations_path):
    if os.path.exists(gloss_dict_file):
        return True

    print(f'Generating gloss_dict at {gloss_dict_file}...')
    try:
        train_corpus = pd.read_csv(os.path.join(annotations_path, 'train.corpus.csv'),
                                   sep='|', header=0, index_col='id')
        test_corpus = pd.read_csv(os.path.join(annotations_path, 'test.corpus.csv'),
                                  sep='|', header=0, index_col='id')
        dev_corpus = pd.read_csv(os.path.join(annotations_path, 'dev.corpus.csv'),
                                 sep='|', header=0, index_col='id')
    except Exception as e:
        print(f"Error reading corpus files: {e}")
        return False

    # get sentences
    sentences = (train_corpus['annotation'].tolist()
                 + test_corpus['annotation'].tolist()
                 + dev_corpus['annotation'].tolist())

    # get sorted glosses
    glosses = sorted({gloss for sentence in sentences for gloss in sentence.split(' ') if gloss})

    # make dict
    gloss_dict = {gloss: i + 1 for i, gloss in enumerate(glosses)}
    assert len(gloss_dict) == len(glosses)

    # save dict
    np.save(gloss_dict_file, gloss_dict)

    # check dict file, if True, return True, else, return False
    return os.path.exists(gloss_dict_file)


def generate_ground_truth(ground_truth_file, annotations_path, mode):
    if os.path.exists(ground_truth_file):
        return True

    print(f'Generating {mode} ground truth at {ground_truth_file}...')

    try:
        corpus = pd.read_csv(os.path.join(annotations_path, f'{mode}.corpus.csv'),
                             sep='|', header=0, index_col='id')
        if mode == 'train':
            corpus.drop('13April_2011_Wednesday_tagesschau_default-14', axis=0, inplace=True)
        with open(ground_truth_file, "w") as f:
            for item in corpus.iterrows():
                f.writelines(
                    f'{item[0]} 1 {item[1]["signer"]} 0.0 1.79769e+308 {item[1]["annotation"]}\n')
    except Exception as e:
        print(f"Error generating ground truth: {e}")
        return False

    return os.path.exists(ground_truth_file)
