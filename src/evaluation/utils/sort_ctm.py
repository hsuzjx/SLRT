import subprocess

from .read_file import read_file
from .write_file import write_file


def sort_ctm(input_file, output_file):
    """
    将输入文件中的内容按照每行的第一列和第三列进行排序。
    
    :param input_file: 输入文件的路径。
    :param output_file: 输出文件的路径。
    :return: 无
    """

    # Sort CTM file
    ctm_sort_cmd = [
        "sort",
        "-k1,1",
        "-k3,3",
        input_file,
        "-o", output_file
    ]
    try:
        subprocess.run(ctm_sort_cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error executing command: {e.cmd}")
        raise
