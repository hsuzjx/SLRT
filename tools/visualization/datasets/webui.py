import argparse

import gradio as gr

from tools.visualization.datasets.CSLDailyVisualizer import CSLDailyVisualizer
from tools.visualization.datasets.Phoenix2014TVisualizer import Phoenix2014TVisualizer
from tools.visualization.datasets.Phoenix2014Visualizer import Phoenix2014Visualizer
from tools.visualization.datasets.utils import TempDirManager


class DatasetVisualizerWebUI:
    def __init__(
            self,
            csl_daily_data_dir: str,
            csl_daily_keypoints_file: str,
            phoenix14_data_dir: str,
            phoenix14_keypoints_file: str,
            phoenix14T_data_dir: str,
            phoenix14T_keypoints_file: str,
            temp_manager: TempDirManager
    ):
        self.temp_manager = temp_manager
        self.temp_dir = self.temp_manager.get_temp_dir()

        self.visualizers = {
            "CSL Daily": CSLDailyVisualizer(
                data_dir=csl_daily_data_dir,
                keypoints_file=csl_daily_keypoints_file,
                temp_dir=self.temp_dir
            ),
            "Phoenix14": Phoenix2014Visualizer(
                data_dir=phoenix14_data_dir,
                keypoints_file=phoenix14_keypoints_file,
                temp_dir=self.temp_dir
            ),
            "Phoenix14T": Phoenix2014TVisualizer(
                dataset_dir=phoenix14T_data_dir,
                keypoints_file=phoenix14T_keypoints_file,
                temp_dir=self.temp_dir
            )
        }

        self.color_dict = {
            "red": (0, 0, 255),
            "green": (0, 255, 0),
            "blue": (255, 0, 0)
        }

    def gr_fn(
            self,
            dataset_name: str,
            sample_idx: str,
            fps: int,
            confidence_threshold: float,
            add_keypoints: bool,
            add_edges: bool,
            add_heatmap: bool,
            keypoints_style: str,
            keypoints_color: str,
            keypoints_scale: int,
            keypoints_thickness: int,
            edges_color: str,
            edges_thickness: int,
            heatmap_sigma: int,
            heatmap_alpha: float,
            heatmap_gamma: float
    ):
        visualizer = self.visualizers[dataset_name]
        visualizer.init_fn(
            fps=fps,
            confidence_threshold=confidence_threshold,
            keypoints_color=self.color_dict[keypoints_color],
            keypoints_scale=keypoints_scale,
            keypoints_style=keypoints_style,
            keypoints_thickness=keypoints_thickness,
            edges_color=self.color_dict[edges_color],
            edges_thickness=edges_thickness,
            heatmap_sigma=heatmap_sigma,
            heatmap_alpha=heatmap_alpha,
            heatmap_gamma=heatmap_gamma,
            add_keypoints=add_keypoints,
            add_edges=add_edges,
            add_heatmap=add_heatmap
        )
        video, glosses, translation, info = visualizer.interface_fn(sample_idx)
        return video, glosses, translation, info

    def launch(self, server_name, server_port, share=False, inbrowser=False):
        iface = gr.Interface(
            title="Sign Language Datasets Visualizer",
            fn=self.gr_fn,
            inputs=[
                # Dataset options
                gr.Dropdown(label="Dataset", choices=["CSL Daily", "Phoenix14", "Phoenix14T"]),
                gr.Text(label='Sample Index',
                        info=f"Sample name or index\nCSL Daily: 0-{self.visualizers['CSL Daily'].dataset_size - 1}, Phoenix14: 0-{self.visualizers['Phoenix14'].dataset_size - 1}, Phoenix14T: 0-{self.visualizers['Phoenix14T'].dataset_size - 1}",
                        value="0"),
                gr.Slider(label="fps", info='Video frame rate', minimum=1, maximum=100, step=1, value=25),
                gr.Slider(label='Confidence Threshold', minimum=0, maximum=1, step=0.05, value=0.3),
                gr.Checkbox(label='Add Keypoint?', value=False),
                gr.Checkbox(label='Add Edge?', value=False),
                gr.Checkbox(label='Add Heatmap?', value=False),
                # Keypoint options
                gr.Radio(label="Keypoint Style", choices=["*", "+", "O", "@", "#", "$", "Number"], value="Number"),
                gr.Radio(label="Keypoint Color", choices=["red", "green", "blue"], value="red"),
                gr.Slider(label='Keypoint Scale', minimum=0, maximum=2, step=0.1, value=0.5),
                gr.Slider(label='Keypoint Thickness', minimum=1, maximum=5, step=1, value=1),
                # Edge options
                gr.Radio(label="Edge Color", choices=["red", "green", "blue"], value="blue"),
                gr.Slider(label='Edge Thickness', minimum=1, maximum=5, step=1, value=1),
                # Heatmap options
                gr.Slider(label='Heatmap Sigma', info='Gaussian kernel size', minimum=1, maximum=50, step=1, value=10),
                gr.Slider(label='Heatmap Alpha', info='Heatmap alpha', minimum=0, maximum=1, step=0.05, value=0.5),
                gr.Slider(label='Heatmap Gamma', info='Heatmap gamma', minimum=-255, maximum=255, step=0.05, value=0),
            ],
            outputs=[
                gr.Video(label='Video', autoplay=True, show_download_button=True),
                gr.Text(label='Gloss Sequence'),
                gr.Text(label='Translation'),
                gr.Text(label='Info')
            ],
            live=False,
            flagging_mode='never',
        )

        iface.launch(
            server_name=server_name,
            server_port=server_port,
            share=share,
            inbrowser=inbrowser,
            allowed_paths=[self.temp_dir]
        )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Sign Language Datasets Visualizer")

    # dataset options
    parser.add_argument("--csl-daily-data-dir", type=str,
                        default="../../../data/csl-daily",
                        help="Path to the CSL Daily dataset directory")
    parser.add_argument("--csl-daily-keypoints-file", type=str,
                        default="../../../data/keypoints/csl-daily/sentence_frames-512x512/frames_512x512/csl-daily-keypoints.pkl",
                        help="Path to the CSL Daily keypoints file")
    parser.add_argument("--phoenix14-data-dir", type=str,
                        default="../../../data/phoenix2014",
                        help="Path to the PHOENIX14 dataset directory")
    parser.add_argument("--phoenix14-keypoints-file", type=str,
                        default="../../../data/keypoints/phoenix2014/fullFrame-210x260px/phoenix2014-keypoints.pkl",
                        help="Path to the PHOENIX14 keypoints file")
    parser.add_argument("--phoenix14T-data-dir", type=str,
                        default="../../../data/phoenix2014T",
                        help="Path to the PHOENIX14T dataset directory")
    parser.add_argument("--phoenix14T-keypoints-file", type=str,
                        default="../../../data/keypoints/phoenix2014T/fullFrame-210x260px/phoenix2014t-keypoints.pkl",
                        help="Path to the PHOENIX14T keypoints file")

    parser.add_argument("--temp-dir", type=str,
                        default="../../../.tmp/ds_vis",
                        help="Path to the temporary directory")
    parser.add_argument("--keep-temp", action="store_true",
                        help="Keep temporary files")
    parser.add_argument("--cleanup-interval", type=int,
                        default=600,
                        help="Cleanup interval in seconds (only applies if --keep-temp is not set)")

    # launch options
    parser.add_argument("--host", type=str, default="localhost", help="Server name")
    parser.add_argument("--port", type=int, default=28081, help="Server port")
    parser.add_argument("--share", action="store_true", help="Share the visualization")
    parser.add_argument("--inbrowse", action="store_true", help="Open browser on launch")

    args = parser.parse_args()

    temp_manager = TempDirManager(
        temp_dir=args.temp_dir,
        cleanup=not args.keep_temp,
        cleanup_interval=args.cleanup_interval
    )

    webui = DatasetVisualizerWebUI(
        csl_daily_data_dir=args.csl_daily_data_dir,
        csl_daily_keypoints_file=args.csl_daily_keypoints_file,
        phoenix14_data_dir=args.phoenix14_data_dir,
        phoenix14_keypoints_file=args.phoenix14_keypoints_file,
        phoenix14T_data_dir=args.phoenix14T_data_dir,
        phoenix14T_keypoints_file=args.phoenix14T_keypoints_file,
        temp_manager=temp_manager
    )

    webui.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        inbrowser=args.inbrowse
    )
