import numpy as np
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean

def dtw_align(visual_features, text_features):
    """
    使用DTW对齐视觉特征和文本特征。
1. 动态时间规整（Dynamic Time Warping, DTW）
DTW 是一种用于测量两个序列之间的相似度的方法，特别适用于长度不一致的序列。它通过在时间轴上进行非线性对齐来最小化两个序列之间的距离
    Args:
        visual_features (np.ndarray): 视觉特征，形状为 (sequence_length, feature_dim)。
        text_features (np.ndarray): 文本特征，形状为 (sequence_length, feature_dim)。

    Returns:
        np.ndarray: 对齐后的视觉特征。
    """
    distance, path = fastdtw(visual_features, text_features, dist=euclidean)
    aligned_visual_features = np.zeros_like(text_features)
    
    for i, j in path:
        aligned_visual_features[j] += visual_features[i]
    
    # 归一化
    aligned_visual_features /= np.sum(aligned_visual_features, axis=0, keepdims=True)
    
    return aligned_visual_features
