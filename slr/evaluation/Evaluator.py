import os
import shutil
import subprocess
import sys
from datetime import datetime

import fcntl
from omegaconf import DictConfig

from slr.evaluation.utils import clean_phoenix_2014, clean_phoenix_2014_trans

DatasetCleanFunctionDict = {
    "phoenix2014": clean_phoenix_2014,
    "phoenix2014T": clean_phoenix_2014_trans,
}


class Evaluator:
    """
    Evaluation class for processing and evaluating hypothesis against ground truth files.

    This class provides methods for preprocessing hypothesis files, running external evaluation tools,
    and cleaning up temporary files. It supports different datasets and can handle various input formats
    for ground truth files.

    Attributes:
        gt_file (dict or str): Ground truth file path or dictionary of paths for 'train', 'dev', and 'test' modes.
        sclite_bin (str): Absolute path to the SCLITE binary used for evaluation.
        dataset (str, optional): Identifier for the dataset being evaluated.
        cleanup (bool): Flag indicating whether to clean up temporary files after evaluation.
    """

    def __init__(self, gt_file, sclite_bin="sclite", dataset=None, cleanup=True):
        """
        Initializes the evaluator with the given parameters.

        Validates input parameters and sets up necessary attributes.

        Args:
            gt_file (dict or str): Ground truth file path or dictionary of paths for 'train', 'dev', and 'test' modes.
            sclite_bin (str, optional): Absolute path to the SCLITE binary used for evaluation.
            dataset (str, optional): Identifier for the dataset being evaluated.
            cleanup (bool, optional): Flag indicating whether to clean up temporary files after evaluation.
        """
        self.sclite_bin = os.path.abspath(sclite_bin)
        self.dataset = dataset
        self.cleanup = cleanup

        # Process gt_file based on its type
        if isinstance(gt_file, dict) or isinstance(gt_file, DictConfig):
            self.gt_file = {mode: os.path.abspath(gt_file.get(mode, None)) for mode in ['train', 'dev', 'test']}
        elif isinstance(gt_file, str):
            self.gt_file = {mode: os.path.abspath(gt_file) for mode in ['train', 'dev', 'test']}
        else:
            raise ValueError("gt_file must be either a dictionary with 'train', 'dev', and 'test' keys or a string.")

    def evaluate(self, save_dir, hyp_file, lock_file, mode):
        """
        Evaluates the hypothesis file against the ground truth using SCLITE.

        Prepares the hypothesis file, runs SCLITE, and extracts the Word Error Rate (WER).

        Args:
            save_dir (str): Directory where results will be saved.
            hyp_file (str): Path to the hypothesis file.
            lock_file (str): Path to the lock file used during preprocessing.
            mode (str): Mode of the dataset ('train', 'dev', or 'test').

        Returns:
            float: The calculated Word Error Rate (WER).
        """
        gt_file = self.gt_file[mode]
        ctm_file = self.preprocess_hyp(save_dir, hyp_file, lock_file, gt_file=gt_file)

        results_dir = os.path.join(save_dir, "sclite_results")
        os.makedirs(results_dir, exist_ok=True)

        # Run SCLITE evaluation
        sclite_args = [
            self.sclite_bin,
            "-h", ctm_file, "ctm",  # Hypothesis file
            "-r", gt_file, "stm",  # Reference file
            "-f", "0",  # Input file format
            "-o", "sgml", "sum", "rsum", "pra", "dtl",  # Output formats
        ]
        if os.path.exists(results_dir):
            sclite_args.extend(["-O", results_dir])

        try:
            # Execute SCLITE with prepared arguments
            print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} Running SCLITE...")
            subprocess.run(sclite_args, capture_output=True, text=True, check=True)

        except subprocess.CalledProcessError as e:
            # Handle errors during SCLITE execution
            print(f"SCLITE execution failed: {e.stderr}")
            raise
        finally:
            print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} SCLITE completed successfully.",
                  f"Outputs saved to {results_dir}")

        # Extract WER from SCLITE output
        with open(os.path.join(results_dir, f"{os.path.basename(ctm_file)}.dtl"), "r") as file:
            for line in file:
                line = line.strip()
                if "Percent Total Error" in line:
                    wer_line = line
                    break
        word_error_rate = float(wer_line.split("=")[1].split("%")[0])

        return word_error_rate

    def preprocess_hyp(self, save_dir, hyp_file, lock_file, gt_file):
        """
        Preprocesses the hypothesis file by converting it to CTM format, merging it with STM, sorting, and cleaning up.

        Args:
            save_dir (str): Directory where temporary files will be saved.
            hyp_file (str): Path to the hypothesis file.
            lock_file (str): Path to the lock file used during preprocessing.
            gt_file (str): Path to the ground truth file.

        Returns:
            str: Path to the preprocessed CTM file.
        """
        basename = os.path.splitext(os.path.basename(hyp_file))[0]

        tmp_file_1 = os.path.join(save_dir, f"{basename}.tmp1.converted.ctm")
        tmp_file_2 = os.path.join(save_dir, f"{basename}.tmp2.merged.ctm")
        tmp_file_3 = os.path.join(save_dir, f"{basename}.tmp3.sorted.ctm")
        output_file = os.path.join(save_dir, f"{basename}.ctm")

        self.convert_hyp_to_ctm(hyp_file, tmp_file_1, lock_file, clean_sentence=True)
        self.merge_ctm_and_stm(tmp_file_1, gt_file, tmp_file_2)
        self.sort_ctm(tmp_file_2, tmp_file_3)

        self.copy_file(tmp_file_3, output_file)

        if self.cleanup:
            self.remove_tmp_file(tmp_file_1)
            self.remove_tmp_file(tmp_file_2)
            self.remove_tmp_file(tmp_file_3)

        return output_file

    def convert_hyp_to_ctm(self, hyp_file, ctm_file, lock_file, clean_sentence=True):
        """
        Converts the hypothesis file to CTM format, optionally cleaning sentences.

        Args:
            hyp_file (str): Path to the hypothesis file.
            ctm_file (str): Path to the output CTM file.
            lock_file (str): Path to the lock file used during conversion.
            clean_sentence (bool, optional): Whether to clean sentences using dataset-specific functions.
        """
        with open(lock_file, 'w') as f:
            # Acquire file lock
            fcntl.flock(f, fcntl.LOCK_EX)
            try:
                with open(hyp_file, "r") as hyp_f:
                    with open(ctm_file, "w") as ctm_f:
                        for line in hyp_f:
                            line_split = line.strip().split()
                            name = line_split[0]
                            sentence = " ".join(line_split[1:])
                            if clean_sentence and self.dataset in DatasetCleanFunctionDict:
                                clean_func = DatasetCleanFunctionDict[self.dataset]
                                sentence = clean_func(sentence)

                            for word_idx, word in enumerate(sentence.split()):
                                ctm_f.write(
                                    f"{name} 1 {word_idx * 1.0 / 100:.2f} {(word_idx + 1) * 1.0 / 100:.2f} {word}\n")
            finally:
                # Release file lock
                fcntl.flock(f, fcntl.LOCK_UN)

    def merge_ctm_and_stm(self, ctm_file, stm_file, merged_ctm_file):
        """
        Merges CTM and STM files, ensuring all keys from STM are included in CTM.

        Args:
            ctm_file (str): Path to the CTM file.
            stm_file (str): Path to the STM file.
            merged_ctm_file (str): Path to the output merged CTM file.
        """
        ctm_lines = [line.split() for line in self.read_file(ctm_file)]
        stm_lines = [line.split() for line in self.read_file(stm_file)]

        ctm_keys = {line[0] for line in ctm_lines}
        stm_keys = {line[0] for line in stm_lines}

        for key in stm_keys:
            if key not in ctm_keys:
                ctm_lines.append([key, "1", "0.000", "0.030", "[EMPTY]"])

        self.write_file(merged_ctm_file, [" ".join(line) for line in ctm_lines])

    def sort_ctm(self, ctm_file, sorted_ctm_file):
        # Sort CTM file by the first and third columns
        ctm_sort_cmd = [
            "sort",
            "-k1,1",
            "-k3,3",
            ctm_file,
            "-o", sorted_ctm_file
        ]
        try:
            subprocess.run(ctm_sort_cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error executing command: {e.cmd}")
            raise

    def remove_tmp_file(self, tmp_file):
        # Remove the specified temporary file
        os.remove(tmp_file)

    @staticmethod
    def read_file(file_name):
        # Read lines from the specified file, stripping whitespace
        if not os.path.isfile(file_name):
            print(f"Error: File {file_name} does not exist.")
            sys.exit(1)
        try:
            with open(file_name, "r") as file:
                return [line.strip() for line in file if line.strip()]
        except IOError as e:
            print(f"Error reading {file_name}: {e}")
            sys.exit(1)

    def copy_file(self, src, dest):
        # Copy a file from source to destination
        shutil.copyfile(src, dest)

    @staticmethod
    def write_file(file_name, data):
        # Write the provided data to the specified file
        try:
            with open(file_name, "w+") as file:
                for line in data:
                    file.write(line + "\n")
        except IOError as e:
            print(f"Error writing {file_name}: {e}")
            sys.exit(1)
