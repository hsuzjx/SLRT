import cv2


class VideoWriter:
    """
    视频写入器类，用于将视频帧写入到一个视频文件中。
    """

    def __init__(self, output_file, fps=25, fourcc='mp4v'):
        """
        初始化视频写入器实例。

        参数:
        - output_file: 字符串，输出视频文件的路径。
        - fps: int，视频的帧率，默认为25。
        - fourcc: 字符串，指定视频编解码器，默认为'mp4v'。
        """
        if not output_file:
            raise ValueError("Output file path cannot be empty.")
        if fps <= 0:
            raise ValueError("FPS must be a positive number.")

        self.output_file = output_file
        self.fps = fps
        self.frame_size = None
        self.writer = None
        self.fourcc = fourcc

    def init_writer(self, frame):
        """
        初始化视频写入器，根据给定的帧设置帧大小并创建视频写入器实例。

        参数:
        - frame: numpy数组，视频的第一帧，用于确定视频的帧大小。
        """
        if self.writer is not None:
            return

        try:
            self.frame_size = (frame.shape[1], frame.shape[0])
            fourcc = cv2.VideoWriter_fourcc(*self.fourcc)
            self.writer = cv2.VideoWriter(self.output_file, fourcc, self.fps, self.frame_size)
        except Exception as e:
            print(f"Error initializing video writer: {e}")
            self.writer = None

    def add_frame(self, frame):
        """
        向视频中添加一帧。

        参数:
        - frame: numpy数组，要添加到视频中的帧。
        """
        if self.writer is None:
            self.init_writer(frame)

        try:
            self.writer.write(frame)
        except Exception as e:
            print(f"Error writing frame to video: {e}")

    def release_writer(self):
        """
        释放视频写入器资源并关闭视频文件。
        """
        if self.writer is not None:
            self.writer.release()
            self.writer = None
