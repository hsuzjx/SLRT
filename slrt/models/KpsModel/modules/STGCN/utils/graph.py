import numpy as np


class Graph():
    """ The Graph to model the skeletons extracted by the openpose

    Args:
        strategy (string): must be one of the follow candidates
        - uniform: Uniform Labeling
        - distance: Distance Partitioning
        - spatial: Spatial Configuration
        For more information, please refer to the section 'Partition Strategies'
            in our paper (https://arxiv.org/abs/1801.07455).

        layout (string): must be one of the follow candidates
        - openpose: Is consists of 18 joints. For more information, please
            refer to https://github.com/CMU-Perceptual-Computing-Lab/openpose#output
        - ntu-rgb+d: Is consists of 25 joints. For more information, please
            refer to https://github.com/shahroudy/NTURGB-D

        max_hop (int): the maximal distance between two connected nodes
        dilation (int): controls the spacing between the kernel points

    """

    def __init__(self,
                 layout='openpose',
                 strategy='uniform',
                 max_hop=1,
                 dilation=1):
        self.max_hop = max_hop
        self.dilation = dilation

        self.get_edge(layout)
        self.hop_dis = get_hop_distance(
            self.num_node, self.edge, max_hop=max_hop)
        self.get_adjacency(strategy)

    def __str__(self):
        return self.A

    def get_edge(self, layout):
        if layout == 'openpose':
            self.num_node = 18
            self_link = [(i, i) for i in range(self.num_node)]
            neighbor_link = [
                (4, 3), (3, 2), (7, 6), (6, 5), (13, 12), (12, 11),
                (10, 9), (9, 8), (11, 5), (8, 2), (5, 1), (2, 1),
                (0, 1), (15, 0), (14, 0), (17, 15), (16, 14)
            ]
            self.edge = self_link + neighbor_link
            self.center = 1
        elif layout == 'ntu-rgb+d':
            self.num_node = 25
            self_link = [(i, i) for i in range(self.num_node)]
            neighbor_1base = [
                (1, 2), (2, 21), (3, 21), (4, 3), (5, 21),
                (6, 5), (7, 6), (8, 7), (9, 21), (10, 9),
                (11, 10), (12, 11), (13, 1), (14, 13), (15, 14),
                (16, 15), (17, 1), (18, 17), (19, 18), (20, 19),
                (22, 23), (23, 8), (24, 25), (25, 12)
            ]
            neighbor_link = [(i - 1, j - 1) for (i, j) in neighbor_1base]
            self.edge = self_link + neighbor_link
            self.center = 21 - 1
        elif layout == 'ntu_edge':
            self.num_node = 24
            self_link = [(i, i) for i in range(self.num_node)]
            neighbor_1base = [
                (1, 2), (3, 2), (4, 3), (5, 2), (6, 5), (7, 6),
                (8, 7), (9, 2), (10, 9), (11, 10), (12, 11),
                (13, 1), (14, 13), (15, 14), (16, 15), (17, 1),
                (18, 17), (19, 18), (20, 19), (21, 22), (22, 8),
                (23, 24), (24, 12)
            ]
            neighbor_link = [(i - 1, j - 1) for (i, j) in neighbor_1base]
            self.edge = self_link + neighbor_link
            self.center = 2
        # elif layout=='customer settings'
        #     pass
        elif layout=="body":
            self.num_node = 11
            self_link = [(i, i) for i in range(self.num_node)]
            neighbor_1base = [
                [6, 7], [6, 8], [7, 9],
                [8, 10], [9, 11], [2, 3], [1, 2], [1, 3],
                [2, 4], [3, 5], [4, 6], [5, 7]
            ]
            neighbor_link = [(i - 1, j - 1) for (i, j) in neighbor_1base]
            self.edge = self_link + neighbor_link
            self.center = 1
        elif layout=="left_hand":
            self.num_node = 21
            self_link = [(i, i) for i in range(self.num_node)]
            neighbor_1base = [
                [92, 93], [92, 97], [92, 101], [92, 105], [92, 109],  # root
                [93, 94], [94, 95], [95, 96],  # finger 1
                [97, 98], [98, 99], [99, 100],  # finger 2
                [101, 102], [102, 103], [103, 104],  # finger 3
                [105, 106], [106, 107], [107, 108],  # finger 4
                [109, 110], [110, 111], [111, 112]  # finger 5
            ]
            neighbor_link = [(i - 92, j - 92) for (i, j) in neighbor_1base]
            self.edge = self_link + neighbor_link
            self.center = 1
        elif layout == "right_hand":
            self.num_node = 21
            self_link = [(i, i) for i in range(self.num_node)]
            neighbor_1base = [
                [113, 114], [113, 118], [113, 122], [113, 126], [113, 130],  # root
                [114, 115], [115, 116], [116, 117],  # finger 1
                [118, 119], [119, 120], [120, 121],  # finger 2
                [122, 123], [123, 124], [124, 125],  # finger 3
                [126, 127], [127, 128], [128, 129],  # finger 4
                [130, 131], [131, 132], [132, 133]  # finger 5
            ]
            neighbor_link = [(i - 113, j - 113) for (i, j) in neighbor_1base]
            self.edge = self_link + neighbor_link
            self.center = 1
        elif layout == "face":
            self.num_node = 68
            self_link = [(i, i) for i in range(self.num_node)]
            neighbor_1base = [
                [24, 25], [25, 26], [26, 27], [27, 28], [28, 29], [29, 30], [30, 31], [31, 32],
                [32, 33], [33, 34], [34, 35], [35, 36], [36, 37], [37, 38], [38, 39], [39, 40],  # facial contour
                [41, 42], [42, 43], [43, 44], [44, 45],  # right eyebrow
                [46, 47], [47, 48], [48, 49], [49, 50],  # left eyebrow
                [51, 52], [52, 53], [53, 54], [54, 55], [54, 59], [55, 56],
                [56, 57], [57, 58], [58, 59], [51, 55], [51, 59],  # nose
                [60, 61], [61, 62], [62, 63], [63, 64], [64, 65], [65, 60],  # right eye
                [66, 67], [67, 68], [68, 69], [69, 70], [70, 71], [71, 66],  # left eye
                # [72, 73], [73, 74], [74, 75], [75, 76], [76, 77], [77, 78],
                # [78, 79], [79, 80], [80, 81], [81, 82], [82, 83], [83, 72],
                # [72, 84], [84, 85], [85, 86], [86, 87], [87, 88], [88, 78],
                # [78, 89], [89, 90], [90, 91], [91, 72]  # mouth
            ]
            neighbor_link = [(i - 24, j - 24) for (i, j) in neighbor_1base]
            self.edge = self_link + neighbor_link
            self.center = 1
        else:
            raise ValueError("Do Not Exist This Layout.")

    def get_adjacency(self, strategy):
        valid_hop = range(0, self.max_hop + 1, self.dilation)
        adjacency = np.zeros((self.num_node, self.num_node))
        for hop in valid_hop:
            adjacency[self.hop_dis == hop] = 1
        normalize_adjacency = normalize_digraph(adjacency)

        if strategy == 'uniform':
            A = np.zeros((1, self.num_node, self.num_node))
            A[0] = normalize_adjacency
            self.A = A
        elif strategy == 'distance':
            A = np.zeros((len(valid_hop), self.num_node, self.num_node))
            for i, hop in enumerate(valid_hop):
                A[i][self.hop_dis == hop] = normalize_adjacency[self.hop_dis ==
                                                                hop]
            self.A = A
        elif strategy == 'spatial':
            A = []
            for hop in valid_hop:
                a_root = np.zeros((self.num_node, self.num_node))
                a_close = np.zeros((self.num_node, self.num_node))
                a_further = np.zeros((self.num_node, self.num_node))
                for i in range(self.num_node):
                    for j in range(self.num_node):
                        if self.hop_dis[j, i] == hop:
                            if self.hop_dis[j, self.center] == self.hop_dis[
                                i, self.center]:
                                a_root[j, i] = normalize_adjacency[j, i]
                            elif self.hop_dis[j, self.
                                    center] > self.hop_dis[i, self.
                                    center]:
                                a_close[j, i] = normalize_adjacency[j, i]
                            else:
                                a_further[j, i] = normalize_adjacency[j, i]
                if hop == 0:
                    A.append(a_root)
                else:
                    A.append(a_root + a_close)
                    A.append(a_further)
            A = np.stack(A)
            self.A = A
        else:
            raise ValueError("Do Not Exist This Strategy")


def get_hop_distance(num_node, edge, max_hop=1):
    A = np.zeros((num_node, num_node))
    for i, j in edge:
        A[j, i] = 1
        A[i, j] = 1

    # compute hop steps
    hop_dis = np.zeros((num_node, num_node)) + np.inf
    transfer_mat = [np.linalg.matrix_power(A, d) for d in range(max_hop + 1)]
    arrive_mat = (np.stack(transfer_mat) > 0)
    for d in range(max_hop, -1, -1):
        hop_dis[arrive_mat[d]] = d
    return hop_dis


def normalize_digraph(A):
    Dl = np.sum(A, 0)
    num_node = A.shape[0]
    Dn = np.zeros((num_node, num_node))
    for i in range(num_node):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i] ** (-1)
    AD = np.dot(A, Dn)
    return AD


def normalize_undigraph(A):
    Dl = np.sum(A, 0)
    num_node = A.shape[0]
    Dn = np.zeros((num_node, num_node))
    for i in range(num_node):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i] ** (-0.5)
    DAD = np.dot(np.dot(Dn, A), Dn)
    return DAD
