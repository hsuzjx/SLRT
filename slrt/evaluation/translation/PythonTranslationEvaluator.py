import os

from omegaconf import DictConfig

from .metrics import bleu, rouge


class PythonTranslationEvaluator(object):
    def __init__(self, gt_file, dataset=None):
        self.dataset = dataset

        # Process gt_file based on its type
        if isinstance(gt_file, dict) or isinstance(gt_file, DictConfig):
            self.gt_file = {mode: os.path.abspath(gt_file.get(mode, None)) for mode in ['train', 'dev', 'test']}
        elif isinstance(gt_file, str):
            self.gt_file = {mode: os.path.abspath(gt_file) for mode in ['train', 'dev', 'test']}
        else:
            raise ValueError("gt_file must be either a dictionary with 'train', 'dev', and 'test' keys or a string.")

    def evaluate(self, save_dir, hyp_file, mode):
        save_dir = os.path.abspath(save_dir)
        hyp_file = os.path.abspath(hyp_file)

        gt_file = os.path.abspath(self.gt_file[mode])

        with open(gt_file, 'r') as f:
            gt_lines = f.readlines()
        ref = {line.split()[0]: ' '.join(line.split()[5:]).strip() for line in gt_lines}

        with open(hyp_file, 'r') as f:
            hyp_lines = f.readlines()
        hyp = {line.split()[0]: ' '.join(line.split()[1:]).strip() for line in hyp_lines}

        ref_list, hyp_list = [], []
        for key in ref.keys():
            ref_list.append(ref[key])
            hyp_list.append(hyp[key] if key in hyp.keys() else "")

        bleu_score_dict = bleu(ref_list, hyp_list, level='char')
        rouge_score = rouge(ref_list, hyp_list, level='char')

        with open(os.path.join(save_dir, "evaluation_results.txt"), 'w') as f:
            f.write(f"ground_truth_file: {gt_file}\n")
            f.write(f"hypothesis_file: {hyp_file}\n\n")
            for k in bleu_score_dict.keys():
                f.write(f"{k}: {bleu_score_dict[k]}\n")
            f.write(f"\nROUGE: {rouge_score}\n")

        return (
            bleu_score_dict['bleu1'],
            bleu_score_dict['bleu2'],
            bleu_score_dict['bleu3'],
            bleu_score_dict['bleu4'],
            rouge_score
        )
