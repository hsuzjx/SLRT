import os
import subprocess
from datetime import datetime

from .process_phoenix2014_output import process_phoenix2014_output


def evaluate(ctm_file, gt_file, save_dir="./", sclite_bin="sclite", dataset=None, cleanup=True):
    """
    Evaluates the performance of the model using SCLITE tool for different sign language datasets.

    Args:
        ctm_file (str): Path to the CTM file containing the predicted outputs.
        gt_file (str): Path to the ground truth file.
        save_dir (str, optional): Directory where the results will be saved. Defaults to "./".
        sclite_bin (str, optional): Path to the SCLITE executable. Defaults to "sclite".
        dataset (str, optional): Name of the dataset being evaluated. If None, no preprocessing is done. Defaults to None.
        cleanup (bool, optional): Whether to delete temporary files after processing. Defaults to True.

    Returns:
        float: The Word Error Rate (WER) calculated by SCLITE.
    """
    # Resolve absolute paths for better file path handling
    sclite_bin = os.path.abspath(sclite_bin)
    gt_file = os.path.abspath(gt_file)
    ctm_file = os.path.abspath(ctm_file)
    save_dir = os.path.abspath(save_dir)
    os.makedirs(save_dir, exist_ok=True)

    # Prepare directory for SCLITE results
    results_dir = os.path.join(save_dir, "sclite_results")
    os.makedirs(results_dir, exist_ok=True)

    processed_ctm_file = os.path.join(save_dir, f"processed.{os.path.basename(ctm_file)}")

    # Define a dictionary to map dataset names to preprocessing functions
    preprocessing_funcs = {
        "phoenix2014": lambda ctm, gt, out: process_phoenix2014_output(ctm, gt, out, remove_tmp_file=cleanup),
        # TODO: Add other datasets like phoenix2014T, csl-daily if necessary
    }

    # Preprocess the CTM file for evaluation
    if dataset is not None and dataset in preprocessing_funcs:
        processed_ctm_file = preprocessing_funcs[dataset](ctm_file, gt_file, processed_ctm_file)
    else:
        processed_ctm_file = ctm_file

    # Run SCLITE evaluation
    sclite_args = [
        sclite_bin,
        "-h", processed_ctm_file, "ctm",  # Hypothesis file
        "-r", gt_file, "stm",  # Reference file
        "-f", "0",  # Format of input files
        "-o", "sgml", "sum", "rsum", "pra", "dtl",  # Output format
    ]
    if results_dir is not None:
        sclite_args.extend(["-O", results_dir])

    try:
        # Execute SCLITE with prepared arguments
        print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} Running SCLITE...")
        subprocess.run(sclite_args, capture_output=True, text=True, check=True)
        print(
            f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} SCLITE completed successfully. Outputs saved to {results_dir}")
    except subprocess.CalledProcessError as e:
        # Handle errors during SCLITE execution
        print(f"SCLITE execution failed: {e.stderr}")
        raise

    # Extract WER from SCLITE output
    with open(os.path.join(results_dir, f"{os.path.basename(processed_ctm_file)}.dtl"), "r") as file:
        for line in file:
            line = line.strip()
            if "Percent Total Error" in line:
                wer_line = line
                break
    word_error_rate = float(wer_line.split("=")[1].split("%")[0])

    return word_error_rate
