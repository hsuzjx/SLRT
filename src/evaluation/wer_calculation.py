import os
import subprocess

from .merge_ctm_stm import merge_ctm_stm


def evaluate(file_save_path="./", groundtruth_file=None, ctm_file=None, evaluate_dir=None,
             sclite_path="../../.bin/sclite",
             python_evaluate=False, triplet=False):
    """
    Evaluates the speech recognition results using SCLITE or Python-based evaluation.
    
    :param file_save_path: Prefix for file paths.
    :param groundtruth_file: Ground truth file path.
    :param ctm_file: Output file name.
    :param evaluate_dir: Directory containing evaluation scripts.
    :param sclite_path: Path to the SCLITE executable.
    :param python_evaluate: Whether to use Python for evaluation.
    :param triplet: Whether to perform triplet evaluation.
    :return: WER value as a float.
    """
    # Resolve absolute paths for better file path handling
    sclite_path = os.path.abspath(sclite_path)

    file_save_path = os.path.abspath(file_save_path)
    groundtruth_file = os.path.abspath(groundtruth_file)
    ctm_file = os.path.abspath(ctm_file)

    # Ensure necessary directories exist
    if not os.path.isdir(file_save_path):
        os.makedirs(file_save_path)

    # Prepare directory for SCLITE results
    results_output_dir = os.path.join(file_save_path, "sclite_results")
    if not os.path.isdir(results_output_dir):
        os.makedirs(results_output_dir)

    # Prepare names for processed and merged CTM files
    ctm_file_base_name = os.path.basename(ctm_file)
    processed_ctm_file = os.path.join(file_save_path, f"processed.{ctm_file_base_name}")
    merged_ctm_file = os.path.join(file_save_path, f"merged.{ctm_file_base_name}")
    sorted_ctm_file = os.path.join(file_save_path, f"sorted.{ctm_file_base_name}")

    # Preprocess and sort STM file
    ctm_process_cmd = [
        "bash", os.path.join(evaluate_dir, "ctm_process.sh"),
        ctm_file,
        processed_ctm_file
    ]

    try:
        # Execute CTM processing and sorting commands
        subprocess.run(ctm_process_cmd, check=True)
    except subprocess.CalledProcessError as e:
        # Handle errors during command execution
        print(f"Error executing command: {e.cmd}")
        raise

    # Merge CTM and STM files
    merge_ctm_stm(processed_ctm_file, groundtruth_file, merged_ctm_file)

    # Sort CTM file
    ctm_sort_cmd = [
        "sort",
        "-k1,1",
        "-k3,3",
        merged_ctm_file,
        "-o", sorted_ctm_file
    ]
    try:
        subprocess.run(ctm_sort_cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error executing command: {e.cmd}")
        raise

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
        "-h", sorted_ctm_file, "ctm",  # Hypothesis file
        "-r", groundtruth_file, "stm",  # Reference file
        "-f", "0",  # Format of input files
        "-o", "sgml", "sum", "rsum", "pra", "dtl",  # Output format
    ]
    if results_output_dir is not None:
        sclite_args.extend(["-O", results_output_dir])

    try:
        # Execute SCLITE with prepared arguments
        subprocess.run(sclite_args, capture_output=True, text=True, check=True)
    except subprocess.CalledProcessError as e:
        # Handle errors during SCLITE execution
        print(f"SCLITE execution failed: {e.stderr}")
        raise

    # Extract WER from SCLITE output
    with open(os.path.join(results_output_dir, f"sorted.{ctm_file_base_name}.dtl"), "r") as f:
        for line in f:
            line = line.strip()
            if "Percent Total Error" in line:
                wer_line = line
                break
    wer = float(wer_line.split("=")[1].split("%")[0])

    return wer
