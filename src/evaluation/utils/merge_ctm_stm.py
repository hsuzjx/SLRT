from .read_file import read_file
from .write_file import write_file


def merge_ctm_stm(processed_ctm_file, sorted_stm_file, merged_ctm_file):
    """
    合并CTM（时间对齐的文本）和STM（标准化的文本标记）文件。
    
    如果CTM文件中不存在STM文件中的某些键（标识符），则为这些键添加默认的空条目。
    空条目格式为：键, "1", "0.000", "0.030", "[EMPTY]"。
    
    参数:
    processed_ctm_file (str): 已处理的CTM文件路径。
    sorted_stm_file (str): 已排序的STM文件路径。
    merged_ctm_file (str): 输出的合并CTM文件路径。
    
    返回:
    无
    """
    # 读取CTM文件的每一行数据
    ctm_lines = [line.split() for line in read_file(processed_ctm_file)]
    # 读取STM文件的每一行数据
    stm_lines = [line.split() for line in read_file(sorted_stm_file)]

    # 提取CTM文件中的所有键（第一列数据），用于后续比较
    ctm_keys = {line[0] for line in ctm_lines}
    # 提取STM文件中的所有键（第一列数据），用于后续比较
    stm_keys = {line[0] for line in stm_lines}

    # 遍历STM文件中的所有键
    for key in stm_keys:
        # 如果STM文件中的键不在CTM文件中，则为该键添加一个空条目
        if key not in ctm_keys:
            ctm_lines.append([key, "1", "0.000", "0.030", "[EMPTY]"])

    # 将合并后的CTM数据写入到指定文件
    write_file(merged_ctm_file, [" ".join(line) for line in ctm_lines])
