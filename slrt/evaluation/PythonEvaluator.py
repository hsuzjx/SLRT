import os

import numpy as np
from omegaconf import DictConfig

from .contents import DatasetCleanFunctionDict


class PythonEvaluator(object):
    """
    Python evaluator for SLR
    """

    def __init__(self, gt_file, dataset=None):
        """
        Initializes the evaluator with the given parameters.

        Validates input parameters and sets up necessary attributes.

        Args:
            gt_file (dict or str): Ground truth file path or dictionary of paths for 'train', 'dev', and 'test' modes.
            dataset (str, optional): Identifier for the dataset being evaluated.
        """
        self.dataset = dataset

        # Process gt_file based on its type
        if isinstance(gt_file, dict) or isinstance(gt_file, DictConfig):
            self.gt_file = {mode: os.path.abspath(gt_file.get(mode, None)) for mode in ['train', 'dev', 'test']}
        elif isinstance(gt_file, str):
            self.gt_file = {mode: os.path.abspath(gt_file) for mode in ['train', 'dev', 'test']}
        else:
            raise ValueError("gt_file must be either a dictionary with 'train', 'dev', and 'test' keys or a string.")

        self.WER_COST_DEL = 3
        self.WER_COST_INS = 3
        self.WER_COST_SUB = 4

    def evaluate(self, save_dir, hyp_file, mode) -> float:
        """
        Evaluates the given hypothesis file against the ground truth file for the specified mode.
        """
        save_dir = os.path.abspath(save_dir)
        hyp_file = os.path.abspath(hyp_file)

        gt_file = os.path.abspath(self.gt_file[mode])

        with open(gt_file, 'r') as f:
            gt_lines = f.readlines()
        ref = {line.split()[0]: ' '.join(line.split()[5:]).strip() for line in gt_lines}

        with open(hyp_file, 'r') as f:
            hyp_lines = f.readlines()
        clean_func = DatasetCleanFunctionDict[self.dataset] if self.dataset in DatasetCleanFunctionDict.keys() else None
        if clean_func is None:
            hyp = {line.split()[0]: ' '.join(line.split()[1:]).strip() for line in hyp_lines}
        else:
            hyp = {line.split()[0]: clean_func(' '.join(line.split()[1:]).strip()) for line in hyp_lines}

        ret = self.calc_wer(ref, hyp)

        with open(os.path.join(save_dir, "evaluation_results.txt"), 'w') as f:
            f.write(f"ground truth file:{gt_file}\n")
            f.write(f"hypothesis file:{hyp_file}\n\n")
            for k in ret.keys():
                f.write(f"{k}:{ret[k]}\n")

        return ret["wer"]

    def calc_wer(self, references, hypotheses):
        total_error = total_del = total_ins = total_sub = total_ref_len = 0

        assert len(references) >= len(hypotheses)

        # for r, h in zip(references, hypotheses):
        for k in references.keys():
            r = references[k]
            h = hypotheses[k] if k in hypotheses.keys() else ""

            res = self.wer_single(r=r, h=h)
            total_error += res["num_err"]
            total_del += res["num_del"]
            total_ins += res["num_ins"]
            total_sub += res["num_sub"]
            total_ref_len += res["num_ref"]

        wer = (total_error / total_ref_len) * 100
        del_rate = (total_del / total_ref_len) * 100
        ins_rate = (total_ins / total_ref_len) * 100
        sub_rate = (total_sub / total_ref_len) * 100

        return {
            "wer": wer,
            "del_rate": del_rate,
            "ins_rate": ins_rate,
            "sub_rate": sub_rate,
            "total_err": total_error,
            "total_del": total_del,
            "total_ins": total_ins,
            "total_sub": total_sub,
            "reference_total_length": total_ref_len
        }

    def wer_single(self, r, h):
        r = r.strip().split()
        h = h.strip().split()
        edit_distance_matrix = self.edit_distance(r=r, h=h)
        alignment, alignment_out = self.get_alignment(r=r, h=h, d=edit_distance_matrix)

        num_cor = np.sum([s == "C" for s in alignment])
        num_del = np.sum([s == "D" for s in alignment])
        num_ins = np.sum([s == "I" for s in alignment])
        num_sub = np.sum([s == "S" for s in alignment])
        num_err = num_del + num_ins + num_sub
        num_ref = len(r)

        return {
            "alignment": alignment,
            "alignment_out": alignment_out,
            "num_cor": num_cor,
            "num_del": num_del,
            "num_ins": num_ins,
            "num_sub": num_sub,
            "num_err": num_err,
            "num_ref": num_ref,
        }

    def edit_distance(self, r, h):
        """
        Original Code from https://github.com/zszyellow/WER-in-python/blob/master/wer.py
        This function is to calculate the edit distance of reference sentence and the hypothesis sentence.
        Main algorithm used is dynamic programming.
        Attributes:
            r -> the list of words produced by splitting reference sentence.
            h -> the list of words produced by splitting hypothesis sentence.
        """
        d = np.zeros((len(r) + 1) * (len(h) + 1), dtype=np.uint8).reshape(
            (len(r) + 1, len(h) + 1)
        )
        for i in range(len(r) + 1):
            for j in range(len(h) + 1):
                if i == 0:
                    # d[0][j] = j
                    d[0][j] = j * self.WER_COST_INS
                elif j == 0:
                    d[i][0] = i * self.WER_COST_DEL
        for i in range(1, len(r) + 1):
            for j in range(1, len(h) + 1):
                if r[i - 1] == h[j - 1]:
                    d[i][j] = d[i - 1][j - 1]
                else:
                    substitute = d[i - 1][j - 1] + self.WER_COST_SUB
                    insert = d[i][j - 1] + self.WER_COST_INS
                    delete = d[i - 1][j] + self.WER_COST_DEL
                    d[i][j] = min(substitute, insert, delete)
        return d

    def get_alignment(self, r, h, d):
        """
        Original Code from https://github.com/zszyellow/WER-in-python/blob/master/wer.py
        This function is to get the list of steps in the process of dynamic programming.
        Attributes:
            r -> the list of words produced by splitting reference sentence.
            h -> the list of words produced by splitting hypothesis sentence.
            d -> the matrix built when calculating the editing distance of h and r.
        """
        x = len(r)
        y = len(h)
        max_len = 3 * (x + y)

        alignlist = []
        align_ref = ""
        align_hyp = ""
        alignment = ""

        while True:
            if (x <= 0 and y <= 0) or (len(alignlist) > max_len):
                break
            elif x >= 1 and y >= 1 and d[x][y] == d[x - 1][y - 1] and r[x - 1] == h[y - 1]:
                align_hyp = " " + h[y - 1] + align_hyp
                align_ref = " " + r[x - 1] + align_ref
                alignment = " " * (len(r[x - 1]) + 1) + alignment
                alignlist.append("C")
                x = max(x - 1, 0)
                y = max(y - 1, 0)
            elif x >= 1 and y >= 1 and d[x][y] == d[x - 1][y - 1] + self.WER_COST_SUB:
                ml = max(len(h[y - 1]), len(r[x - 1]))
                align_hyp = " " + h[y - 1].ljust(ml) + align_hyp
                align_ref = " " + r[x - 1].ljust(ml) + align_ref
                alignment = " " + "S" + " " * (ml - 1) + alignment
                alignlist.append("S")
                x = max(x - 1, 0)
                y = max(y - 1, 0)
            elif y >= 1 and d[x][y] == d[x][y - 1] + self.WER_COST_INS:
                align_hyp = " " + h[y - 1] + align_hyp
                align_ref = " " + "*" * len(h[y - 1]) + align_ref
                alignment = " " + "I" + " " * (len(h[y - 1]) - 1) + alignment
                alignlist.append("I")
                x = max(x, 0)
                y = max(y - 1, 0)
            else:
                align_hyp = " " + "*" * len(r[x - 1]) + align_hyp
                align_ref = " " + r[x - 1] + align_ref
                alignment = " " + "D" + " " * (len(r[x - 1]) - 1) + alignment
                alignlist.append("D")
                x = max(x - 1, 0)
                y = max(y, 0)

        align_ref = align_ref[1:]
        align_hyp = align_hyp[1:]
        alignment = alignment[1:]

        return (
            alignlist[::-1],
            {"align_ref": align_ref, "align_hyp": align_hyp, "alignment": alignment},
        )
