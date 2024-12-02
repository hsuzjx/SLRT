from typing_extensions import override

from slrt.datasets.Datasets.KeypointDatasets.KeypointBaseDataset import KeypointBaseDataset


class Phoenix2014TKeypointDataset(KeypointBaseDataset):
    """
    """

    def __init__(
            self,
            keypoints_file: str = None,
            mode: [str, list] = "train",
            transform: callable = None,
            tokenizer: object = None,
            frame_size: tuple = (210, 260)
    ):
        """
        """
        super().__init__(keypoints_file=keypoints_file, transform=transform, tokenizer=tokenizer, frame_size=frame_size)

        # Convert mode to list and validate
        self.mode = [mode] if isinstance(mode, str) else mode
        if not all(m in ["train", "dev", "test"] for m in self.mode):
            raise ValueError("Each element in mode must be one of 'train', 'dev', or 'test'")

        self.kps_info_keys = sorted([n for n in self.kps_info.keys() if self.kps_info[n]["split"] in self.mode])

    @override
    def _get_glosses(self, item) -> [str, list]:
        return [gloss for gloss in item['orth'].split(' ') if gloss]

    @override
    def _get_translation(self, item) -> [str, list]:
        return [word for word in item['translation'].split(' ') if word]
