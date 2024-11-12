import glob
import os.path
import pickle
from concurrent.futures import ThreadPoolExecutor

import cv2
import pandas as pd
from tqdm import tqdm


# 定义一个函数，用于调整Phoenix2014数据集的图片大小
def resize_phoenix2014_dataset(feature_path, annotations_path, saved_path, dsize=(255, 255)):
    """
    调整Phoenix2014数据集的图片大小。
    
    参数:
    feature_path (str): 原始图片路径。
    annotations_path (str): 注释文件路径。
    saved_path (str): 调整大小后的图片保存路径。
    dsize (tuple, 可选): 目标图片大小，默认为(255, 255)。
    """
    # 将路径转换为绝对路径
    feature_path = os.path.abspath(feature_path)
    annotations_path = os.path.abspath(annotations_path)
    saved_path = os.path.abspath(saved_path)

    # 确保保存路径存在，如果不存在则创建
    os.makedirs(saved_path, exist_ok=True)

    # 遍历不同的数据集模式
    for mode in ['train', 'dev', 'test']:
        # 读取注释文件中的数据
        annotations = pd.read_csv(os.path.join(annotations_path, '{}.corpus.csv'.format(mode)), sep='|', header=0,
                                  index_col='id')

        # 遍历每个文件夹
        for folder in annotations['folder']:
            # 获取文件夹中所有图片的路径
            file_list = glob.glob(os.path.join(feature_path, mode, folder))
            # 遍历每张图片
            for file in file_list:
                # 读取图片
                img = cv2.imread(file)
                # 调整图片大小
                img_resized = cv2.resize(img, dsize, interpolation=cv2.INTER_LANCZOS4)
                # 构建保存路径
                saved_file = os.path.join(saved_path, mode, os.path.dirname(folder), os.path.basename(file))
                # 确保保存路径的目录存在，如果不存在则创建
                os.makedirs(os.path.dirname(saved_file), exist_ok=True)
                # 保存调整大小后的图片
                cv2.imwrite(saved_file, img_resized)
                # 输出保存的文件路径
                print(saved_file)


def resize_csl_daily(dataset_dir, features_output_dir, max_workers=4):
    dataset_dir = os.path.abspath(dataset_dir)
    features_output_dir = os.path.abspath(features_output_dir)
    os.makedirs(features_output_dir, exist_ok=True)

    features_dir = os.path.join(dataset_dir, 'sentence_frames-512x512/frames_512x512')
    annotations_dir = os.path.join(dataset_dir, 'sentence_label')

    if not os.path.exists(features_dir):
        raise FileNotFoundError(f"Features directory not found at {features_dir}")
    if not os.path.exists(annotations_dir):
        raise FileNotFoundError(f"Annotations directory not found at {annotations_dir}")

    annotation_file = os.path.join(annotations_dir, "csl2020ct_v2.pkl")
    if not os.path.exists(annotation_file):
        raise FileNotFoundError(f"Annotation file not found at {annotation_file}")
    with open(annotation_file, 'rb') as f:
        data = pickle.load(f)
    info = pd.DataFrame(data['info'])

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for idx, row in tqdm(info.iterrows(), total=info.shape[0], desc='Get futures'):
            frames_dir = os.path.join(features_dir, row['name'])
            if not os.path.exists(frames_dir):
                raise FileNotFoundError(f"Frames directory not found at {frames_dir}")
            file_list = glob.glob(os.path.join(frames_dir, '*.jpg'))
            output_subdir = os.path.join(features_output_dir, row['name'])
            for file in file_list:
                futures.append(executor.submit(resize_image, file, output_subdir, (256, 256)))

        # 等待所有任务完成
        for future in tqdm(futures, total=len(futures), desc='Saving resized images'):
            future.result()


def resize_phoenix2014T(dataset_dir, features_output_dir, max_workers=4):
    dataset_dir = os.path.abspath(dataset_dir)
    features_output_dir = os.path.abspath(features_output_dir)
    os.makedirs(features_output_dir, exist_ok=True)

    features_dir = os.path.join(dataset_dir, 'PHOENIX-2014-T/features/fullFrame-210x260px')
    annotations_dir = os.path.join(dataset_dir, 'PHOENIX-2014-T/annotations/manual')

    if not os.path.exists(features_dir):
        raise FileNotFoundError(f"Features directory not found at {features_dir}")
    if not os.path.exists(annotations_dir):
        raise FileNotFoundError(f"Annotations directory not found at {annotations_dir}")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for mode in ['train', 'dev', 'test']:
            annotation_file = os.path.join(annotations_dir, f"PHOENIX-2014-T.{mode}.corpus.csv")
            if not os.path.exists(annotation_file):
                raise FileNotFoundError(f"Annotation file not found at {annotation_file}")
            info = pd.read_csv(annotation_file, sep='|', header=0, index_col='name')

            for idx, row in tqdm(info.iterrows(), total=info.shape[0], desc=f'Get futures, mode-{mode}'):
                frames_dir = os.path.join(features_dir, mode, idx)
                if not os.path.exists(frames_dir):
                    raise FileNotFoundError(f"Frames directory not found at {frames_dir}")
                file_list = glob.glob(os.path.join(frames_dir, '*.png'))
                output_subdir = os.path.join(features_output_dir, mode, idx)
                for file in file_list:
                    futures.append(executor.submit(resize_image, file, output_subdir, (256, 256)))

        # 等待所有任务完成
        for future in tqdm(futures, total=len(futures), desc=f'Saving resized images'):
            future.result()


def resize_image(file, output_dir, dsize=(256, 256)):
    img = cv2.imread(file)
    img_resized = cv2.resize(img, dsize, interpolation=cv2.INTER_LANCZOS4)
    saved_file = os.path.join(output_dir, os.path.basename(file))
    os.makedirs(os.path.dirname(saved_file), exist_ok=True)
    cv2.imwrite(saved_file, img_resized)


# 当作为主程序运行时，调用函数处理数据集
if __name__ == '__main__':
    # resize_phoenix2014_dataset('../../data/phoenix2014/phoenix-2014-multisigner/features/fullFrame-210x260px',
    #                            '../../data/phoenix2014/phoenix-2014-multisigner/annotations/manual',
    #                            '../../data/phoenix2014/phoenix-2014-multisigner/features/fullFrame-256x256px',
    #                            (256, 256))

    # resize_csl_daily(dataset_dir="/new_home/xzj23/workspace/SLR/data/csl-daily",
    #                  features_output_dir="./csl-daily/frames_256x256",
    #                  max_workers=8)

    resize_phoenix2014T(dataset_dir="/new_home/xzj23/workspace/SLR/data/phoenix2014T",
                        features_output_dir="./phoenix2014T/frames_256x256",
                        max_workers=8)
