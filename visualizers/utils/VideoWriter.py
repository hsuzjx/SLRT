import cv2


class VideoWriter:
    """
    用于创建和管理视频文件的写入。
    
    属性：
    - output_path：字符串，视频文件的输出路径。
    - fps：整数，视频的帧率，默认为25。
    - frame_size：元组，视频帧的尺寸，初始值为None。
    - writer：cv2.VideoWriter对象，用于实际的视频写入操作，初始值为None。
    """

    def __init__(self, output_path, fps=25):
        """
        初始化视频写入器。
        
        参数：
        - output_path：输出视频的文件路径。
        - fps：视频的帧率，默认为25。
        """
        self.output_path = output_path
        self.fps = fps
        self.frame_size = None
        self.writer = None

    def init_writer(self, frame):
        """
        初始化视频写入器，设置帧尺寸和编码器。
        
        参数：
        - frame：numpy数组，视频中的第一帧，用于获取帧尺寸。
        """
        if self.writer is not None:
            return
        self.frame_size = (frame.shape[1], frame.shape[0])
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.writer = cv2.VideoWriter(self.output_path, fourcc, self.fps, self.frame_size)

    def add_frame(self, frame):
        """
        向视频中添加一帧。
        
        参数：
        - frame：numpy数组，要添加到视频中的帧。
        """
        if self.writer is None:
            self.init_writer(frame)
        self.writer.write(frame)

    def release_writer(self):
        """
        释放视频写入器资源。
        """
        if self.writer is not None:
            self.writer.release()
