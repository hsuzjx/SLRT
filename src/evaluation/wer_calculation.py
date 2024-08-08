import os
import subprocess


# from .mergectmstm_1722914048172 import merge_ctm_stm


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

    # Create sorted ground truth file path
    sorted_ground_truth_file = os.path.join(file_save_path, "sorted_ground_truth",
                                            f"sorted.{os.path.basename(groundtruth_file)}")
    # Ensure necessary directories exist
    if not os.path.isdir(file_save_path):
        os.makedirs(file_save_path)
    if not os.path.isdir(os.path.dirname(sorted_ground_truth_file)):
        os.makedirs(os.path.dirname(sorted_ground_truth_file))

    # Prepare directory for SCLITE results
    results_output_dir = os.path.join(file_save_path, "sclite_results")
    if not os.path.isdir(results_output_dir):
        os.makedirs(results_output_dir)

    # Prepare names for processed and merged CTM files
    ctm_file_base_name = os.path.basename(ctm_file)
    processed_ctm_file = os.path.join(file_save_path, f"processed.{ctm_file_base_name}")
    merged_ctm_file = os.path.join(file_save_path, f"merged.{ctm_file_base_name}")

    # Preprocess and sort STM file
    ctm_process_cmd = [
        "bash", os.path.join(evaluate_dir, "ctm_process.sh"),
        ctm_file,
        processed_ctm_file
    ]
    sort_cmd = [
        "sort",
        "-k1,1",
        groundtruth_file,
        "-o", sorted_ground_truth_file
    ]

    try:
        # Execute CTM processing and sorting commands
        subprocess.run(ctm_process_cmd, check=True)
        if not os.path.exists(sorted_ground_truth_file):
            print("Sorted ground truth file not found. Sorting...")
            subprocess.run(sort_cmd, check=True)
            print(f"Sorted ground truth file created. Saved in {sorted_ground_truth_file}")
    except subprocess.CalledProcessError as e:
        # Handle errors during command execution
        print(f"Error executing command: {e.cmd}")
        print(f"Return code: {e.returncode}")
        print(f"Standard output: {e.stdout}")
        print(f"Standard error: {e.stderr}")
        raise

    # Merge CTM and STM files
    # TODO: ...
    # merge_ctm_stm(processed_ctm_file, sorted_ground_truth_file, merged_ctm_file)
    merge_cmd = [
        "python", os.path.join(evaluate_dir, "mergectmstm.py"),
        processed_ctm_file,
        sorted_ground_truth_file
    ]
    subprocess.run(merge_cmd, check=True)

    # Copy merged CTM file to final output location
    subprocess.run(["cp", processed_ctm_file, merged_ctm_file], check=True)

    # TODO: Implement Python-based evaluation
    # if python_evaluate:
    #     ret = wer_calculation(os.path.join(evaluate_dir, f"{evaluate_prefix}-{mode}.stm"), merged_ctm_file)
    #     if triplet:
    #         wer_calculation(
    #             os.path.join(evaluate_dir, f"{evaluate_prefix}-{mode}.stm"),
    #             merged_ctm_file,
    #             merged_ctm_file.replace(".ctm", "-conv.ctm")
    #         )
    #     return ret

    # Run SCLITE evaluation
    sclite_args = [
        sclite_path,
        "-h", merged_ctm_file, "ctm",  # Hypothesis file
        "-r", sorted_ground_truth_file, "stm",  # Reference file
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
    with open(os.path.join(results_output_dir, f"merged.{ctm_file_base_name}.dtl"), "r") as f:
        for line in f:
            line = line.strip()
            if "Percent Total Error" in line:
                wer_line = line
                break
    wer = float(wer_line.split("=")[1].split("%")[0])

    return wer
