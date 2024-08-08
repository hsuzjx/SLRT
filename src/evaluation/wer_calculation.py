import os
import subprocess


def evaluate(prefix="./", mode="dev", evaluate_dir=None, evaluate_prefix=None,
             output_file=None, output_dir=None, python_evaluate=False,
             triplet=False):
    """
    Evaluates the speech recognition results using SCLITE or Python-based evaluation.
    
    :param prefix: Prefix for file paths.
    :param mode: Evaluation mode (e.g., "dev").
    :param evaluate_dir: Directory containing evaluation scripts.
    :param evaluate_prefix: Prefix for evaluation files.
    :param output_file: Output file name.
    :param output_dir: Optional output directory.
    :param python_evaluate: Whether to use Python for evaluation.
    :param triplet: Whether to perform triplet evaluation.
    :return: WER value as a float.
    """
    sclite_path = "../../.bin/sclite"
    sclite_path = os.path.abspath(sclite_path)

    # Preprocess and sort STM file
    preprocess_cmd = [
        "bash", os.path.join(evaluate_dir, "ctm_process.sh"),
        os.path.join(prefix, output_file),
        os.path.join(prefix, f"tmp2.{output_file}"),
        os.path.join(prefix, f"tmp.{output_file}")
    ]
    sort_cmd = [
        "sort",
        "-k1,1",
        os.path.join(evaluate_dir, f"{evaluate_prefix}-{mode}.stm"),
        "-o", os.path.join(prefix, f"tmp.stm")
    ]

    try:
        subprocess.run(preprocess_cmd, check=True)
        subprocess.run(sort_cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error executing command: {e.cmd}")
        raise

    # Merge CTM and STM files
    merge_cmd = [
        "python", os.path.join(evaluate_dir, "mergectmstm.py"),
        os.path.join(prefix, f"tmp2.{output_file}"),
        os.path.join(prefix, f"tmp.stm")
    ]
    subprocess.run(merge_cmd, check=True)

    # Copy merged CTM file to final output location
    out_ctm = os.path.join(prefix, f"out.{output_file}")
    subprocess.run(["cp", os.path.join(prefix, f"tmp2.{output_file}"), out_ctm], check=True)

    # TODO: Implement Python-based evaluation
    # if python_evaluate:
    #     ret = wer_calculation(os.path.join(evaluate_dir, f"{evaluate_prefix}-{mode}.stm"), out_ctm)
    #     if triplet:
    #         wer_calculation(
    #             os.path.join(evaluate_dir, f"{evaluate_prefix}-{mode}.stm"),
    #             out_ctm,
    #             out_ctm.replace(".ctm", "-conv.ctm")
    #         )
    #     return ret

    # Run SCLITE evaluation
    sclite_args = [
        sclite_path,
        "-h", out_ctm, "ctm",
        "-r", os.path.join(prefix, f"tmp.stm"), "stm",
        "-f", "0",
        "-o", "sgml sum rsum pra"
    ]
    if output_dir is not None:
        output_full_dir = os.path.join(prefix, output_dir)
        if not os.path.isdir(output_full_dir):
            os.makedirs(output_full_dir)
        sclite_args.extend(["-O", output_full_dir])

    try:
        result = subprocess.run(sclite_args, capture_output=True, text=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"SCLITE execution failed: {e.stderr}")
        raise

    # Extract WER from SCLITE output
    error_line = [line for line in result.stdout.split('\n') if line.startswith("Error")][0]
    ret = float(error_line.split("=")[1].split("%")[0])

    return ret
