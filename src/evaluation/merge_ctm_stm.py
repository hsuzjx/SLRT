#!/usr/bin/env python

import sys
import os

from datetime import datetime


def read_file(file_name):
    """安全地读取文件，返回各行的列表"""
    if not os.path.isfile(file_name):
        print(f"Error: File {file_name} does not exist.")
        sys.exit(1)
    try:
        with open(file_name, "r") as file:
            return [line.strip().split() for line in file]
    except IOError as e:
        print(f"Error reading {file_name}: {e}")
        sys.exit(1)


def write_file(file_name, data):
    """安全地写入数据到文件"""
    try:
        with open(file_name, "w+") as file:
            for line in data:
                file.write(" ".join(line) + "\n")
    except IOError as e:
        print(f"Error writing {file_name}: {e}")
        sys.exit(1)


def merge_ctm_stm(processed_ctm_file, sorted_stm_file, merged_ctm_file):
    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} Merging CTM and STM files...")
    ctm_lines = read_file(processed_ctm_file)
    stm_lines = read_file(sorted_stm_file)

    ctm_keys = {line[0] for line in ctm_lines}
    stm_keys = {line[0] for line in stm_lines}

    for key in stm_keys:
        if key not in ctm_keys:
            ctm_lines.append([key, "1", "0.000", "0.030", "[EMPTY]"])

    write_file(merged_ctm_file, ctm_lines)
    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} Merged CTM and STM files saved to {merged_ctm_file}")
