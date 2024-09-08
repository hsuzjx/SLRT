import sys


def write_file(file_name, data):
    """安全地写入数据到文件"""
    try:
        with open(file_name, "w+") as file:
            for line in data:
                file.write(line + "\n")
    except IOError as e:
        print(f"Error writing {file_name}: {e}")
        sys.exit(1)
