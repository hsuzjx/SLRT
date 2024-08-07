import os
import numpy as np
import pandas as pd


def preprocess(dataset_name, annotations_path, gloss_dict_path, ground_truth_path):
    """
    根据数据集名称执行相应的预处理函数。

    Parameters:
    - dataset_name: 字符串，表示数据集的名称。
    - annotations_path: 字符串，表示注释文件的路径。
    - gloss_dict_path: 字符串，表示词汇字典保存的路径。
    - ground_truth_path: 字符串，表示地面真实数据保存的路径。

    Returns:
    - 如果预处理成功，返回相应的处理函数的返回值。
    """
    if dataset_name == 'phoenix2014':
        return preprocess_phoenix2014(dataset_name, annotations_path, gloss_dict_path, ground_truth_path)
    elif dataset_name == 'phoenix2014T':
        return preprocess_phoenix2014T(dataset_name, annotations_path, gloss_dict_path, ground_truth_path)


def preprocess_phoenix2014(dataset_name, annotations_path, gloss_dict_path, ground_truth_path):
    """
    对'phoenix2014'数据集进行预处理，包括生成词汇字典和地面真实数据。

    Parameters:
    - dataset_name: 字符串，表示数据集的名称。
    - annotations_path: 字符串，表示注释文件的路径。
    - gloss_dict_path: 字符串，表示词汇字典保存的路径。
    - ground_truth_path: 字符串，表示地面真实数据保存的路径。

    Returns:
    - 如果预处理成功，返回True，否则退出程序。
    """
    # gloss dict
    gloss_dict_file = os.path.join(os.path.abspath(gloss_dict_path), f'{dataset_name}_gloss_dict.npy')
    if not os.path.exists(os.path.dirname(gloss_dict_file)):
        os.makedirs(os.path.dirname(gloss_dict_file))
    if not generate_phoenix2014_gloss_dict(gloss_dict_file, annotations_path):
        exit(1)

    # ground truth
    ground_truth_path = os.path.abspath(ground_truth_path)
    for mode in ['train', 'dev', 'test']:
        ground_truth_file = os.path.join(ground_truth_path, f'{dataset_name}-groundtruth-{mode}.stm')
        if not os.path.exists(os.path.dirname(ground_truth_file)):
            os.makedirs(os.path.dirname(ground_truth_file))
        if not generate_phoenix2014_ground_truth(ground_truth_file, annotations_path, mode):
            exit(1)


def preprocess_phoenix2014T(dataset_name, annotations_path, gloss_dict_path, ground_truth_path):
    """
    对'phoenix2014T'数据集进行预处理，包括生成词汇字典和地面真实数据。

    Parameters:
    - dataset_name: 字符串，表示数据集的名称。
    - annotations_path: 字符串，表示注释文件的路径。
    - gloss_dict_path: 字符串，表示词汇字典保存的路径。
    - ground_truth_path: 字符串，表示地面真实数据保存的路径。

    Returns:
    - 如果预处理成功，返回True，否则退出程序。
    """
    # gloss dict
    gloss_dict_file = os.path.join(os.path.abspath(gloss_dict_path), f'{dataset_name}_gloss_dict.npy')
    if not os.path.exists(os.path.dirname(gloss_dict_file)):
        os.makedirs(os.path.dirname(gloss_dict_file))
    if not generate_phoenix2014T_gloss_dict(gloss_dict_file, annotations_path):
        exit(1)

    # ground truth
    ground_truth_path = os.path.abspath(ground_truth_path)
    for mode in ['train', 'dev', 'test', 'train-complex-annotation']:
        ground_truth_file = os.path.join(ground_truth_path, f'{dataset_name}-groundtruth-{mode}.stm')
        if not os.path.exists(os.path.dirname(ground_truth_file)):
            os.makedirs(os.path.dirname(ground_truth_file))
        if not generate_phoenix2014T_ground_truth(ground_truth_file, annotations_path, mode):
            exit(1)


def _generate_gloss_dict(gloss_dict_file: str, annotations_path: str, filename_pattern, column_name, index_col_name):
    """
    Generates a gloss dictionary from annotation files if it doesn't exist.

    :param gloss_dict_file: File path to save the gloss dictionary.
    :param annotations_path: Path to the annotation files.
    :param filename_pattern: Pattern for annotation file names, formatted with mode (train, test, dev).
    :param column_name: Name of the column containing annotations in the datasets.
    :param index_col_name: Name of the index column in the datasets.
    :return: True if the gloss dictionary file is successfully generated or already exists, False otherwise.
    """
    if os.path.exists(gloss_dict_file):
        return True

    print(f'Generating gloss_dict at {gloss_dict_file}...')
    try:
        datasets = []
        for mode in ['train', 'test', 'dev']:
            dataset = pd.read_csv(os.path.join(annotations_path, filename_pattern.format(mode)), sep='|', header=0,
                                  index_col=index_col_name)
            datasets.append(dataset)

        # get sentences
        sentences = [line[1][column_name] for dataset in datasets for line in dataset.iterrows()]

        # get sorted glosses
        glosses = sorted({gloss for sentence in sentences for gloss in sentence.split(' ') if gloss})

        # make dict
        gloss_dict = {gloss: i + 1 for i, gloss in enumerate(glosses)}
        assert len(gloss_dict) == len(glosses)

        # save dict
        np.save(gloss_dict_file, gloss_dict)

        # check dict file, if True, return True, else, return False
        return os.path.exists(gloss_dict_file)
    except Exception as e:
        print(f"Error reading corpus files: {e}")
        return False


def generate_phoenix2014_gloss_dict(gloss_dict_file, annotations_path):
    """
    Generates a gloss dictionary for the Phoenix-2014 dataset.

    :param gloss_dict_file: File path to save the gloss dictionary.
    :param annotations_path: Path to the Phoenix-2014 annotation files.
    :return: True if the gloss dictionary file is successfully generated or already exists, False otherwise.
    """
    return _generate_gloss_dict(gloss_dict_file, annotations_path, '{}.corpus.csv', 'annotation', 'id')


def generate_phoenix2014T_gloss_dict(gloss_dict_file, annotations_path):
    """
    Generates a gloss dictionary for the Phoenix-2014-T dataset.

    :param gloss_dict_file: File path to save the gloss dictionary.
    :param annotations_path: Path to the Phoenix-2014-T annotation files.
    :return: True if the gloss dictionary file is successfully generated or already exists, False otherwise.
    """
    # TODO: complex annotation
    # ...
    # TODO: add the processes of translation
    # ...
    return _generate_gloss_dict(gloss_dict_file, annotations_path, 'PHOENIX-2014-T.{}.corpus.csv', 'orth', 'name')


def _generate_ground_truth(ground_truth_file: str, annotations_path: str, file_name, index_col_name, column_names,
                           drop_id=None):
    """
    Generates a ground truth file from annotation files if it doesn't exist.

    :param ground_truth_file: File path to save the ground truth.
    :param annotations_path: Path to the annotation files.
    :param file_name: Name of the annotation file.
    :param index_col_name: Name of the index column in the dataset.
    :param column_names: Dictionary containing column names for 'signer' and 'annotation'.
    :param drop_id: Optional. ID of a row to drop from the ground truth.
    :return: True if the ground truth file is successfully generated or already exists, False otherwise.
    """
    if os.path.exists(ground_truth_file):
        return True

    print(f'Generating ground truth at {ground_truth_file}...')

    # Ensure the directory exists
    os.makedirs(os.path.dirname(ground_truth_file), exist_ok=True)

    try:
        # Load the corpus
        corpus = pd.read_csv(os.path.join(annotations_path, file_name),
                             sep='|', header=0, index_col=index_col_name)

        # Drop specific ID if needed
        if drop_id is not None and drop_id in corpus.index:
            corpus.drop(drop_id, axis=0, inplace=True)

        # Write to file
        with open(ground_truth_file, "w") as f:
            for item in corpus.iterrows():
                line = f"{item[0]} 1 {item[1][column_names['signer']]} 0.0 1.79769e+308 {item[1][column_names['annotation']]}\n"
                f.write(line)
    except FileNotFoundError as e:
        print(f"File not found error: {e}")
        return False
    except pd.errors.EmptyDataError as e:
        print(f"Empty data error: {e}")
        return False
    except Exception as e:
        print(f"Error generating ground truth: {e}")
        return False

    return os.path.exists(ground_truth_file)


def generate_phoenix2014_ground_truth(ground_truth_file, annotations_path, mode):
    """
    Generates a ground truth file for the Phoenix-2014 dataset.

    :param ground_truth_file: File path to save the ground truth.
    :param annotations_path: Path to the Phoenix-2014 annotation files.
    :param mode: Mode of the dataset (train, test, dev).
    :return: True if the ground truth file is successfully generated or already exists, False otherwise.
    """
    file_name = f'{mode}.corpus.csv'
    index_col_name = 'id'
    column_names = {'signer': 'signer', 'annotation': 'annotation'}
    drop_id = '13April_2011_Wednesday_tagesschau_default-14' if mode == 'train' else None
    return _generate_ground_truth(ground_truth_file, annotations_path, file_name, index_col_name, column_names, drop_id)


def generate_phoenix2014T_ground_truth(ground_truth_file, annotations_path, mode):
    """
    Generates a ground truth file for the Phoenix-2014-T dataset.

    :param ground_truth_file: File path to save the ground truth.
    :param annotations_path: Path to the Phoenix-2014-T annotation files.
    :param mode: Mode of the dataset (train, test, dev).
    :return: True if the ground truth file is successfully generated or already exists, False otherwise.
    """
    file_name = f'PHOENIX-2014-T.{mode}.corpus.csv'
    index_col_name = 'name'
    column_names = {'signer': 'speaker', 'annotation': 'orth'}
    return _generate_ground_truth(ground_truth_file, annotations_path, file_name, index_col_name, column_names)
