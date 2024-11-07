import os
import shutil
import tempfile
import threading


class TempDirManager:
    """
    临时目录管理器类，负责清理临时目录，并支持定时重复清理功能。
    """

    def __init__(self, temp_dir):
        """
        初始化TempDirManager类。

        :param temp_dir: 临时目录的路径。
        """
        try:
            if temp_dir is None:
                # 创建一个临时目录
                temp_dir = tempfile.mkdtemp()
                print(f"temp_dir is None, Auto-created temporary directory and set to {temp_dir}")
            else:
                temp_dir = os.path.abspath(temp_dir)
                if os.path.exists(temp_dir):
                    print(f"{temp_dir} is existing, use it directly")
                else:
                    print(f"{temp_dir} does not exist, make dirs")
                    os.makedirs(temp_dir)
                temp_dir = temp_dir

        except Exception as e:
            print(f"Failed to create temporary directory: {e}")
            raise

        self.temp_dir = temp_dir
        self.timer = None

    def get_temp_dir(self):
        return self.temp_dir

    def clean_tmp_dir(self):
        """
        清理临时目录。如果目录不存在或没有权限删除，会打印相应的错误信息。
        """
        try:
            shutil.rmtree(self.temp_dir)
            print(f"Folder {self.temp_dir} has been deleted.")
        except FileNotFoundError:
            print(f"The folder {self.temp_dir} does not exist.")
        except PermissionError:
            print(f"Permission denied when trying to delete {self.temp_dir}.")
        except Exception as e:
            print(f"An error occurred while deleting {self.temp_dir}: {e}")

    def start_repeat_delete(self, interval):
        """
        开始定时重复清理临时目录。

        :param interval: 重复清理的时间间隔（秒）。
        """
        self.stop_repeat_delete()
        self.timer = threading.Timer(interval, self.repeat_delete, args=[interval])
        self.timer.start()

    def stop_repeat_delete(self):
        """
        停止定时重复清理。
        """
        if self.timer:
            self.timer.cancel()
            self.timer = None

    def repeat_delete(self, interval):
        """
        重复清理临时目录的方法。先清理一次临时目录，然后重新安排下一次清理任务。

        :param interval: 重复清理的时间间隔（秒）。
        """
        self.clean_tmp_dir()
        self.start_repeat_delete(interval)
