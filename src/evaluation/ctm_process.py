import os
import re
import sys
from datetime import datetime


def process_ctm(input_file, output_file):
    # 检查输入文件是否存在且可读
    if not os.path.isfile(input_file) or not os.access(input_file, os.R_OK):
        print(f"Error: Input file '{input_file}' does not exist or is not readable.")
        sys.exit(1)

    # 检查输出文件是否已存在
    if os.path.exists(output_file):
        print(f"Warning: Output file '{output_file}' already exists and will be overwritten.")

    # 开始处理前记录时间戳
    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} Starting CTM processing...")

    replacements = {
        r'loc-|cl-|qu-|poss-|lh-|__EMOTION__|__PU__|__LEFTHAND__': '',
        r'S0NNE': 'SONNE',
        r'HABEN2': 'HABEN',
        r'WIE AUSSEHEN': 'WIE-AUSSEHEN',
        r'ZEIGEN ': 'ZEIGEN-BILDSCHIRM ',
        r'ZEIGEN$': 'ZEIGEN-BILDSCHIRM',
        r'^([A-Z]) ([A-Z][+ ])': r'\1+\2',
        r'[ +]([A-Z]) ([A-Z]) ': r' \1+\2 ',
        r'([ +][A-Z]) ([A-Z][ +])': r'\1+\2',
        r'([ +]SCH) ([A-Z][ +])': r'\1+\2',
        r'([ +]NN) ([A-Z][ +])': r'\1+\2',
        r'([ +][A-Z]) (NN[ +])': r'\1+\2',
        r'([ +][A-Z]) ([A-Z])$': r'\1+\2',
        r'([A-Z][A-Z])RAUM': r'\1',
        r'-PLUSPLUS': ''
    }

    with open(input_file, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    # 文本转换
    processed_lines = []
    for line in lines:
        for pattern, replacement in replacements.items():
            line = re.sub(pattern, replacement, line)
        line = re.sub(r'(\b[A-Z]+\b) \1(?![\w-])', r'\1', line)  # 移除重复单词
        line = re.sub(r'__LEFTHAND__|__EPENTHESIS__|__EMOTION__', '', line)  # 过滤特定标记
        line = line.rstrip()  # 清理每行尾部空白
        processed_lines.append(line)

    # 使用awk类似的处理逻辑
    processed_data = []
    last_id = ""
    last_row = ""
    cnt = {}
    for line in processed_lines:
        parts = line.split()
        if len(parts) >= 5:
            id_ = parts[0]
            if last_id != id_ and cnt.get(last_id, 0) < 1 and last_row:
                processed_data.append(last_row + " [EMPTY]")
            if parts[4]:
                cnt[id_] = cnt.get(id_, 0) + 1
                processed_data.append(line)
            last_id = id_
            last_row = line

    # 写入输出文件
    with open(output_file, 'w', encoding='utf-8') as file:
        file.write('\n'.join(processed_data))

    # 记录处理完成时间戳或错误信息
    if os.path.exists(output_file):
        print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} CTM processing finished. Output to {output_file}")
    else:
        print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} Error during CTM processing.")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python script.py <hypothesis-CTM-file> <output-file>")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]

    process_ctm(input_file, output_file)
