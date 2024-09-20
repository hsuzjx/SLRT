import pickle

def safe_pickle_load(file_path):
    """
    安全地从给定的文件路径中加载pickle数据。

    该函数尝试从指定的文件中加载pickle格式的数据，并在加载过程中确保数据安全。
    如果数据不是字典类型，函数将抛出一个ValueError异常，指明数据结构无效。
    在发生任何加载错误（包括文件不存在、数据格式不正确等）时，函数将捕获异常，
    打印错误信息并返回一个空字典。

    参数:
    file_path (str): pickle数据文件的路径。

    返回:
    dict: 加载的pickle数据，如果加载失败则返回空字典。
    """
    try:
        # 打开文件并尝试加载pickle数据
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        # 检查加载的数据是否为字典类型
        if not isinstance(data, dict):
            raise ValueError("Invalid data structure")
        return data
    except Exception as e:
        # 打印加载错误信息
        print(f"Error loading data: {e}")
        return {}
