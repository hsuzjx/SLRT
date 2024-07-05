import os

import numpy as np
import pandas as pd


def preprocess(dataset_name, annotations_path, gloss_dict_path, ground_truth_path):
    # gloss dict
    gloss_dict_path = os.path.abspath(gloss_dict_path)
    if not os.path.exists(gloss_dict_path):
        os.makedirs(gloss_dict_path)
    gloss_dict_file = os.path.join(gloss_dict_path, '{}_gloss_dict.npy'.format(dataset_name))
    if not generate_gloss_dict(gloss_dict_file, annotations_path):
        exit(1)

    # ground truth
    ground_truth_path = os.path.abspath(ground_truth_path)
    if not os.path.exists(ground_truth_path):
        os.makedirs(ground_truth_path)
    for mode in ['train', 'dev', 'test']:
        if not generate_ground_truth(
                os.path.join(ground_truth_path, '{}-groundtruth-{}.stm'.format(dataset_name, mode)),
                annotations_path, mode):
            exit(1)


def generate_gloss_dict(gloss_dict_file, annotations_path):
    if os.path.exists(gloss_dict_file):
        return True

    print('Generating gloss_dict...')
    # get corpus
    train_corpus = pd.read_csv(os.path.join(annotations_path, 'train.corpus.csv'),
                               sep='|', header=0, index_col='id')
    test_corpus = pd.read_csv(os.path.join(annotations_path, 'test.corpus.csv'),
                              sep='|', header=0, index_col='id')
    dev_corpus = pd.read_csv(os.path.join(annotations_path, 'dev.corpus.csv'),
                             sep='|', header=0, index_col='id')

    # get sentences
    sentences = ([s for s in train_corpus.annotation]
                 + [s for s in test_corpus.annotation]
                 + [s for s in dev_corpus.annotation])

    # get sorted glosses
    glosses = set()
    for sentence in sentences:
        for gloss in sentence.split(' '):
            glosses.add(gloss)
    if '' in glosses:
        glosses.remove('')
    glosses = sorted(list(glosses))

    # make dict
    gloss_dict = dict()
    for i in range(len(glosses)):
        gloss_dict[glosses[i]] = i + 1
    assert gloss_dict.__len__() == len(glosses)

    # save dict
    np.save(gloss_dict_file, gloss_dict)

    # check dict file, if True, return True, else, return False
    return os.path.exists(gloss_dict_file)


def generate_ground_truth(ground_truth_file, annotations_path, mode):
    if os.path.exists(ground_truth_file):
        return True

    print('Generating {} ground truth...'.format(mode))

    corpus = pd.read_csv(os.path.join(annotations_path, f'{mode}.corpus.csv'),
                         sep='|', header=0, index_col='id')
    if mode == 'train':
        corpus.drop('13April_2011_Wednesday_tagesschau_default-14', axis=0, inplace=True)
    with open(ground_truth_file, "w") as f:
        for item in corpus.iterrows():
            f.writelines(
                '{} 1 {} 0.0 1.79769e+308 {}\n'.format(item[0], item[1]['signer'], item[1]['annotation']))

    return os.path.exists(ground_truth_file)
