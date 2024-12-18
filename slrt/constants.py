import torch
import transformers
from torchvision.transforms import Compose, RandomCrop, Normalize, RandomHorizontalFlip, CenterCrop

import slrt
from slrt.datasets.transforms import ToTensor, TemporalRescale, RandomDrop
from slrt.datasets.transforms.keypoints import DefinedDorp, RandomMove

CONFIG_PATH = '../configs'
CONFIG_NAME = 'exp_corrnet_csldaily.yaml'

DataModuleClassDict = {
    "phoenix2014": slrt.datasets.DataModules.Phoenix2014DataModule,
    "phoenix2014T": slrt.datasets.DataModules.Phoenix2014TDataModule,
    "csl-daily": slrt.datasets.DataModules.CSLDailyDataModule,
    "phoenix2014-keypoint": slrt.datasets.DataModules.Phoenix2014KeypointDataModule,
    "phoenix2014T-keypoint": slrt.datasets.DataModules.Phoenix2014TKeypointDataModule,
    "csl-daily-keypoint": slrt.datasets.DataModules.CSLDailyKeypointDataModule,
}

ModelClassDict = {
    "CorrNet": slrt.models.CorrNet,
    "SwinBertSLR": slrt.models.SwinBertSLR,
    "MSKA": slrt.models.MSKA,
    "STKA": slrt.models.STKA,
}

InputSampleDict = {
    "CorrNet": (torch.randn(1, 100, 3, 224, 224).to('cpu'), torch.LongTensor([100]).to('cpu')),
    "SLTransformer": None,
    "SwinBertSLR": (
        torch.randn(1, 100, 3, 224, 224).to('cpu'), torch.LongTensor([100]).to('cpu'),
        torch.randn(1, 100, 512), torch.LongTensor([10])
    ),
    "MSKA": (torch.randn(1, 3, 100, 133).to('cpu'), torch.LongTensor([100]).to('cpu')),
    "STKA": (torch.randn(1, 3, 100, 133).to('cpu'), torch.LongTensor([100]).to('cpu')),
}

TokenizerDict = {
    "SimpleTokenizer": slrt.datasets.Tokenizers.SimpleTokenizer,
    "BertTokenizer": transformers.BertTokenizer,
}

DecoderDict = {
    # "CTCBeamSearchDecoder": slrt.models.CTCBeamSearchDecoder,
    "TFCTCBeamSearchDecoder": slrt.models.TFCTCBeamSearchDecoder,
}

EvaluatorDict = {
    "sclite": slrt.evaluation.ScliteEvaluator,
    "python": slrt.evaluation.PythonEvaluator,
}

TransformDict = {
    "video": {
        'train': Compose([ToTensor(), RandomCrop(224), RandomHorizontalFlip(0.5), TemporalRescale(0.2),
                          Normalize(mean=[127.5, 127.5, 127.5], std=[127.5, 127.5, 127.5])]),
        'dev': Compose([ToTensor(), CenterCrop(224), Normalize(mean=[127.5, 127.5, 127.5], std=[127.5, 127.5, 127.5])]),
        'test': Compose([ToTensor(), CenterCrop(224), Normalize(mean=[127.5, 127.5, 127.5], std=[127.5, 127.5, 127.5])])
    },
    "keypoint": {
        'train': Compose([DefinedDorp(0.5, 1.5, 400),
                          # RandomDrop(0.3),
                          RandomMove()]),
        'dev': Compose([DefinedDorp(1, 1, 400)]),
        'test': Compose([DefinedDorp(1, 1, 400)])
    }
}
