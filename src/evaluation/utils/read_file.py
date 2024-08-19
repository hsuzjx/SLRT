import os
import sys


def read_file(file_name):
    """安全地读取文件，返回各行的列表"""
    if not os.path.isfile(file_name):
        print(f"Error: File {file_name} does not exist.")
        sys.exit(1)
    try:
        with open(file_name, "r") as file:
            return [line.strip() for line in file if line.strip()]
    except IOError as e:
        print(f"Error reading {file_name}: {e}")
        sys.exit(1)
