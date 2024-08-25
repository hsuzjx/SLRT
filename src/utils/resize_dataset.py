import glob
import os.path

import cv2
import pandas as pd


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


# 当作为主程序运行时，调用函数处理数据集
if __name__ == '__main__':
    resize_phoenix2014_dataset('../../data/phoenix2014/phoenix-2014-multisigner/features/fullFrame-210x260px',
                               '../../data/phoenix2014/phoenix-2014-multisigner/annotations/manual',
                               '../../data/phoenix2014/phoenix-2014-multisigner/features/fullFrame-256x256px',
                               (256, 256))
