import os
import subprocess
from datetime import datetime

from slr.evaluation.utils import format_phoenix2014_output
from slr.evaluation.utils import merge_ctm_stm, sort_ctm


class Evaluator:
    def __init__(self, save_dir, gt_file, sclite_bin="sclite", dataset=None, cleanup=True):
        """
        初始化Evaluator对象。

        参数:
        - save_dir (str): 结果保存目录的路径。
        - gt_file (dict or str): 地面实况文件的路径。如果是字典，应包含 'dev'、'test' 和 'train' 键；如果是字符串，则所有键都设置为该文件路径。
        - sclite_bin (str, 可选): SCLITE可执行文件的路径，默认为"sclite"。
        - dataset (str, 可选): 被评估的数据集名称，如果未指定，则不进行预处理，默认为None。
        - cleanup (bool, 可选): 是否在处理后删除临时文件，默认为True。
        """
        self.save_dir = os.path.abspath(save_dir)
        self.sclite_bin = os.path.abspath(sclite_bin)
        self.dataset = dataset
        self.cleanup = cleanup

        # 处理gt_file的不同情况
        if isinstance(gt_file, dict):
            self.gt_file = {mode: os.path.abspath(gt_file.get(mode, None)) for mode in ['train', 'dev', 'test']}
        elif isinstance(gt_file, str):
            self.gt_file = {mode: os.path.abspath(gt_file) for mode in ['train', 'dev', 'test']}
        else:
            raise ValueError("gt_file must be either a dictionary with 'dev', 'test', and 'train' keys or a string.")

    def process_phoenix2014_output(self, file, ground_truth_file, processed_file, remove_tmp_file=True):
        """
        Preprocesses the output of the Phoenix 2014 dataset.

        This function modifies, merges, and sorts the CTM files to ensure the output meets the required format standards.

        Parameters:
        - file: The path to the original CTM file.
        - ground_truth_file: The path to the ground truth STM file.
        - processed_file: The path to the final processed CTM file.
        - remove_tmp_file: Whether to delete temporary files after processing.
        """
        # Get the directory and base name of the file for subsequent file operations
        file_save_dir = os.path.dirname(file)
        base_name = os.path.basename(file)

        # Prepare names for processed and merged CTM files
        formated_ctm_file = os.path.join(file_save_dir, f"tmp1.formated.{base_name}")
        merged_ctm_file = os.path.join(file_save_dir, f"tmp2.merged.{base_name}")

        # Output the start time of the preprocessing
        print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} Processing CTM file...")

        # Modify the original CTM file to correct the format
        format_phoenix2014_output(file, formated_ctm_file)

        # Merge CTM and STM files to ensure completeness of the transcription
        merge_ctm_stm(formated_ctm_file, ground_truth_file, merged_ctm_file)

        # Sort the merged CTM file to facilitate subsequent processing
        sort_ctm(merged_ctm_file, processed_file)

        # Optionally delete temporary files
        if remove_tmp_file:
            os.remove(formated_ctm_file)
            os.remove(merged_ctm_file)

        # Output completion time and processed file path
        print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} Processing CTM file done. Output to {processed_file}")

    def evaluate(self, ctm_file, mode='dev'):
        """
        使用SCLITE工具评估模型性能。

        参数:
        - ctm_file (str): 包含预测输出的CTM文件路径。
        - mode (str, 可选): 指定使用的地面实况文件键，默认为'dev'。

        返回:
        - float or None: 由SCLITE计算出的词错误率(WER)，如果地面实况文件不存在则返回None。
        """
        # 解析绝对路径以更好地处理文件路径
        ctm_file = os.path.abspath(ctm_file)
        os.makedirs(self.save_dir, exist_ok=True)

        # 确定当前使用的地面实况文件
        if mode not in ['dev', 'test', 'train']:
            raise ValueError("mode must be one of 'dev', 'test', or 'train'.")

        gt_file = self.gt_file.get(mode, None)

        # 如果地面实况文件不存在，则跳过评估并返回None
        if gt_file is None:
            print(
                f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} Ground truth file for mode '{mode}' does not exist. Skipping evaluation.")
            return None

        # 准备SCLITE结果目录
        results_dir = os.path.join(self.save_dir, "sclite_results")
        os.makedirs(results_dir, exist_ok=True)

        processed_ctm_file = os.path.join(self.save_dir, f"processed.{os.path.basename(ctm_file)}")

        # # 定义一个字典来映射数据集名称到预处理函数
        # preprocessing_funcs = {
        #     "phoenix2014": lambda ctm, gt, out: process_phoenix2014_output(ctm, gt, out, remove_tmp_file=self.cleanup),
        #     # TODO: 如有必要，添加其他数据集如phoenix2014T, csl-daily
        # }

        # 预处理CTM文件以进行评估
        if self.dataset is not None and self.dataset == "phoenix2014":
            self.process_phoenix2014_output(ctm_file, gt_file, processed_ctm_file)
        else:
            processed_ctm_file = ctm_file

        # 运行SCLITE评估
        sclite_args = [
            self.sclite_bin,
            "-h", processed_ctm_file, "ctm",  # 假设文件
            "-r", gt_file, "stm",  # 参考文件
            "-f", "0",  # 输入文件格式
            "-o", "sgml", "sum", "rsum", "pra", "dtl",  # 输出格式
        ]
        if results_dir is not None:
            sclite_args.extend(["-O", results_dir])

        try:
            # 执行SCLITE带有准备好的参数
            print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} Running SCLITE...")
            subprocess.run(sclite_args, capture_output=True, text=True, check=True)
            print(
                f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} SCLITE completed successfully. Outputs saved to {results_dir}")
        except subprocess.CalledProcessError as e:
            # 处理SCLITE执行期间的错误
            print(f"SCLITE execution failed: {e.stderr}")
            raise

        # 从SCLITE输出中提取WER
        with open(os.path.join(results_dir, f"{os.path.basename(processed_ctm_file)}.dtl"), "r") as file:
            for line in file:
                line = line.strip()
                if "Percent Total Error" in line:
                    wer_line = line
                    break
        word_error_rate = float(wer_line.split("=")[1].split("%")[0])

        return word_error_rate
