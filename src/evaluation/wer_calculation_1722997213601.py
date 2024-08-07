import os
import subprocess
import tempfile

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
    sclite_path = "./software/sclite"
    
    with tempfile.TemporaryDirectory(dir=prefix) as temp_dir:
        tmp_ctm = os.path.join(temp_dir, 'tmp.ctm')
        tmp_stm = os.path.join(temp_dir, 'tmp.stm')
        
        # Preprocess and sort STM file
        preprocess_cmd = [
            f"bash {evaluate_dir}/preprocess.sh",
            f"{prefix}{output_file}",
            tmp_ctm,
            os.path.join(temp_dir, 'tmp2.ctm')
        ]
        sort_cmd = [
            "sort",
            "-k1,1",
            f"{evaluate_dir}/{evaluate_prefix}-{mode}.stm"
        ]
        
        subprocess.run(preprocess_cmd, check=True)
        subprocess.run(sort_cmd, stdout=open(tmp_stm, 'w'), check=True)

        # Merge CTM and STM files
        merge_cmd = [
            "python",
            f"{evaluate_dir}/mergectmstm.py",
            os.path.join(temp_dir, 'tmp2.ctm'),
            tmp_stm
        ]
        subprocess.run(merge_cmd, check=True)

        # Copy merged CTM file to final output location
        out_ctm = os.path.join(prefix, f"out.{output_file}")
        subprocess.run(["cp", os.path.join(temp_dir, 'tmp2.ctm'), out_ctm], check=True)

        if python_evaluate:
            ret = wer_calculation(f"{evaluate_dir}/{evaluate_prefix}-{mode}.stm", out_ctm)
            if triplet:
                wer_calculation(
                    f"{evaluate_dir}/{evaluate_prefix}-{mode}.stm",
                    out_ctm,
                    out_ctm.replace(".ctm", "-conv.ctm")
                )
            return ret
        
        # Run SCLITE evaluation
        sclite_args = [
            sclite_path,
            "-h", out_ctm, "ctm",
            "-r", tmp_stm, "stm",
            "-f", "0",
            "-o", "sgml sum rsum pra"
        ]
        if output_dir is not None:
            if not os.path.isdir(prefix + output_dir):
                os.makedirs(prefix + output_dir)
            sclite_args.extend(["-O", prefix + output_dir])
        
        result = subprocess.run(sclite_args, capture_output=True, text=True, check=True)
        
        # Extract WER from SCLITE output
        error_line = [line for line in result.stdout.split('\n') if line.startswith("Error")][0]
        ret = float(error_line.split("=")[1].split("%")[0])
        
    return ret
