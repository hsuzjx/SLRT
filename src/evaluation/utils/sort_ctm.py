from .read_file import read_file
from .write_file import write_file


def sort_ctm(input_file, output_file):
    """
    将输入文件中的内容按照每行的第一列和第三列进行排序。
    
    :param input_file: 输入文件的路径。
    :param output_file: 输出文件的路径。
    :return: 无
    """
    # 读取输入文件的内容
    lines = read_file(input_file)

    # 根据每行的第一列和第三列对行进行排序
    sorted_lines = sorted(lines, key=lambda x: (x.split()[0], x.split()[2]))

    # 将排序后的行写入到输出文件中
    write_file(output_file, sorted_lines)
