#!/usr/bin/env python

import sys
import os

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

def main(ctm_file, stm_file):
    ctm_lines = read_file(ctm_file)
    stm_lines = read_file(stm_file)

    ctm_dict = {line[0]: line for line in ctm_lines}
    stm_dict = {line[0]: line for line in stm_lines}

    added_lines = 0
    for key in stm_dict:
        if key not in ctm_dict:
            ctm_lines.insert(list(ctm_dict.keys()).index(key) + added_lines, [key, "1 0.000 0.030 [EMPTY]"])
            added_lines += 1

    write_file(ctm_file, ctm_lines)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: script.py <ctmFile> <stmFile>")
        sys.exit(1)
    ctmFile = sys.argv[1]
    stmFile = sys.argv[2]
    main(ctmFile, stmFile)
