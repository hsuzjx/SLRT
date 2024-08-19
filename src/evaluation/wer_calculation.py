import os
import subprocess
from datetime import datetime

from .process_phoenix2014_output import process_phoenix2014_output


def evaluate(dataset_name, file_save_path="./", ground_truth_file=None, ctm_file=None, sclite_path="../../.bin/sclite",
             remove_tmp_file=True,
             python_evaluate=False, triplet=False):
    """
    Evaluates the speech recognition results using SCLITE or Python-based evaluation.
    
    :param dataset_name: Name of the dataset for which the evaluation is performed.
    :param file_save_path: Prefix for file paths.
    :param ground_truth_file: Ground truth file path.
    :param ctm_file: Output file name.
    :param sclite_path: Path to the SCLITE executable.
    :param remove_tmp_file: Whether to remove temporary files after processing.
    :param python_evaluate: Whether to use Python for evaluation.
    :param triplet: Whether to perform triplet evaluation.
    :return: WER value as a float.
    """
    # Resolve absolute paths for better file path handling
    sclite_path = os.path.abspath(sclite_path)

    ground_truth_file = os.path.abspath(ground_truth_file)
    ctm_file = os.path.abspath(ctm_file)

    file_save_path = os.path.abspath(file_save_path)
    os.makedirs(file_save_path, exist_ok=True)

    # Prepare directory for SCLITE results
    results_output_dir = os.path.join(file_save_path, "sclite_results")
    os.makedirs(results_output_dir, exist_ok=True)

    processed_ctm_file = os.path.join(file_save_path, f"processed.{os.path.basename(ctm_file)}")

    # Preprocess the CTM file for evaluation
    if dataset_name == "phoenix2014":
        process_phoenix2014_output(ctm_file, ground_truth_file, processed_ctm_file, remove_tmp_file=remove_tmp_file)
    elif dataset_name == "phoenix2014T":
        # TODO: Implement preprocessing for Phoenix 2014T
        pass
    elif dataset_name == "csl-daily":
        # TODO: Implement preprocessing for CSL-Daily
        pass
    else:
        # TODO: ...
        raise ValueError("Invalid dataset name. Supported datasets are .....")

    # TODO: Implement Python-based evaluation
    # if python_evaluate:
    #     ret = wer_calculation(os.path.join(evaluate_dir, f"{evaluate_prefix}-{mode}.stm"), sorted_ctm_file)
    #     if triplet:
    #         wer_calculation(
    #             os.path.join(evaluate_dir, f"{evaluate_prefix}-{mode}.stm"),
    #             sorted_ctm_file,
    #             sorted_ctm_file.replace(".ctm", "-conv.ctm")
    #         )
    #     return ret

    # Run SCLITE evaluation
    sclite_args = [
        sclite_path,
        "-h", processed_ctm_file, "ctm",  # Hypothesis file
        "-r", ground_truth_file, "stm",  # Reference file
        "-f", "0",  # Format of input files
        "-o", "sgml", "sum", "rsum", "pra", "dtl",  # Output format
    ]
    if results_output_dir is not None:
        sclite_args.extend(["-O", results_output_dir])

    try:
        # Execute SCLITE with prepared arguments
        print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} Running SCLITE...")
        subprocess.run(sclite_args, capture_output=True, text=True, check=True)
        print(
            f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} SCLITE completed successfully. Outputs saved to {results_output_dir}")
    except subprocess.CalledProcessError as e:
        # Handle errors during SCLITE execution
        print(f"SCLITE execution failed: {e.stderr}")
        raise

    # Extract WER from SCLITE output
    with open(os.path.join(results_output_dir, f"{os.path.basename(processed_ctm_file)}.dtl"), "r") as f:
        for line in f:
            line = line.strip()
            if "Percent Total Error" in line:
                wer_line = line
                break
    wer = float(wer_line.split("=")[1].split("%")[0])

    return wer
