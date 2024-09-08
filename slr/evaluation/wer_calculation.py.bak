import os

from .python_wer_evaluation import wer_calculation


def evaluate(mode, sh_path='./.tmp', save_path='./.tmp', ground_truth_path='./.tmp', mer_path='./evaluation/slr_eval',
             triplet=False, use_python_wer=False
             ):
    os.system(
        f"bash {sh_path}/preprocess.sh {save_path + '/output-hypothesis-{}.ctm'.format(mode)} {save_path}/tmp.ctm {save_path}/tmp2.ctm")
    os.system(f"cat {ground_truth_path}/phoenix2014-groundtruth-{mode}.stm | sort  -k1,1 > {save_path}/tmp.stm")
    # tmp2.ctm: prediction result; tmp.stm: ground-truth result
    os.system(f"python {mer_path}/mergectmstm.py {save_path}/tmp2.ctm {save_path}/tmp.stm")
    os.system(f"cp {save_path}/tmp2.ctm {save_path}/out.output-hypothesis-{mode}.ctm")

    if use_python_wer:
        ret = wer_calculation(f"{ground_truth_path}/phoenix2014-groundtruth-{mode}.stm",
                              f"{save_path}/out.output-hypothesis-{mode}.ctm")
        # if triplet:
        #     wer_calculation(f"{ground_truth_path}/phoenix2014-groundtruth-{mode}.stm",
        #                     f"{save_path}/out.output-hypothesis-{mode}.ctm",
        #                     f"{save_path}/out.output-hypothesis-{mode}.ctm".replace(".ctm", "-conv.ctm")
        #                     )

        # if triplet:
        # wer_calculation(
        #         f"{ground_truth_path}/phoenix2014-groundtruth-{mode}.stm",
        #         f"{save_path}/out.output-hypothesis-{mode}.ctm",
        #         f"{save_path}/out.output-hypothesis-{mode}.ctm".replace(".ctm", "-conv.ctm")
        #     )
    else:
        if not os.path.exists(save_path + '/sclite_results'):
            os.makedirs(save_path + '/sclite_results')
        sclite_path = os.path.abspath('../.bin/sclite')
        os.system(
            f"{sclite_path} -h {save_path}/out.output-hypothesis-{mode}.ctm ctm"
            f" -r {save_path}/tmp.stm stm -f 0 -o sgml sum rsum pra -O {save_path}/sclite_results"
        )
        ret = os.popen(
            f"{sclite_path}  -h {save_path}/out.output-hypothesis-{mode}.ctm ctm "
            f"-r {save_path}/tmp.stm stm -f 0 -o dtl stdout |grep Error"
        ).readlines()[0]
    return ret


def evaluate_old(prefix="./", mode="dev", evaluate_dir=None, evaluate_prefix=None,
                 output_file=None, output_dir=None, python_evaluate=False,
                 triplet=False):
    '''
    TODO  change file save path
    '''
    sclite_path = "./software/sclite"
    print(os.getcwd())
    os.system(f"bash {evaluate_dir}/preprocess.sh {prefix + output_file} {prefix}tmp.ctm {prefix}tmp2.ctm")
    os.system(f"cat {evaluate_dir}/{evaluate_prefix}-{mode}.stm | sort  -k1,1 > {prefix}tmp.stm")
    # tmp2.ctm: prediction result; tmp.stm: ground-truth result
    os.system(f"python {evaluate_dir}/mergectmstm.py {prefix}tmp2.ctm {prefix}tmp.stm")
    os.system(f"cp {prefix}tmp2.ctm {prefix}out.{output_file}")
    if python_evaluate:
        ret = wer_calculation(f"{evaluate_dir}/{evaluate_prefix}-{mode}.stm", f"{prefix}out.{output_file}")
        if triplet:
            wer_calculation(
                f"{evaluate_dir}/{evaluate_prefix}-{mode}.stm",
                f"{prefix}out.{output_file}",
                f"{prefix}out.{output_file}".replace(".ctm", "-conv.ctm")
            )
        return ret
    if output_dir is not None:
        if not os.path.isdir(prefix + output_dir):
            os.makedirs(prefix + output_dir)
        os.system(
            f"{sclite_path}  -h {prefix}out.{output_file} ctm"
            f" -r {prefix}tmp.stm stm -f 0 -o sgml sum rsum pra -O {prefix + output_dir}"
        )
    else:
        os.system(
            f"{sclite_path}  -h {prefix}out.{output_file} ctm"
            f" -r {prefix}tmp.stm stm -f 0 -o sgml sum rsum pra"
        )
    ret = os.popen(
        f"{sclite_path}  -h {prefix}out.{output_file} ctm "
        f"-r {prefix}tmp.stm stm -f 0 -o dtl stdout |grep Error"
    ).readlines()[0]
    return float(ret.split("=")[1].split("%")[0])


if __name__ == "__main__":
    evaluate("output-hypothesis-dev.ctm", mode="dev")
    evaluate("output-hypothesis-test.ctm", mode="test")
