import torch
import transformers
from torchvision.transforms import Compose, RandomCrop, Normalize, RandomHorizontalFlip, CenterCrop

import slr.datasets
import slr.models
from slr.datasets.transforms import ToTensor, TemporalRescale

CONFIG_PATH = '../configs'
CONFIG_NAME = 'CorrNet_Phoenix2014T_experiment.yaml'

DataModuleClassDict = {
    "phoenix2014": slr.datasets.DataModules.Phoenix2014DataModule,
    "phoenix2014T": slr.datasets.DataModules.Phoenix2014TDataModule,
    "csl-daily": slr.datasets.DataModules.CSLDailyDataModule,
    "phoenix2014-keypoint": slr.datasets.DataModules.Phoenix2014KeypointDataModule,
    "phoenix2014T-keypoint": slr.datasets.DataModules.Phoenix2014TKeypointDataModule,
    "csl-daily-keypoint": slr.datasets.DataModules.CSLDailyKeypointDataModule,
}

ModelClassDict = {
    "CorrNet": slr.models.CorrNet,
    "SwinBertSLR": slr.models.SwinBertSLR,
    "MSKA": slr.models.MSKA,
}

InputSampleDict = {
    "CorrNet": (torch.randn(1, 100, 3, 224, 224).to('cpu'), torch.LongTensor([100]).to('cpu')),
    "SLTransformer": None,
    "SwinBertSLR": (
        torch.randn(1, 100, 3, 224, 224).to('cpu'), torch.LongTensor([100]).to('cpu'),
        torch.randn(1, 100, 512), torch.LongTensor([10])
    ),
    "MSKA": ()
}

TokenizerDict = {
    "SimpleTokenizer": slr.datasets.Tokenizers.SimpleTokenizer,
    "BertTokenizer": transformers.BertTokenizer,
}

DecoderDict = {
    "CTCBeamSearchDecoder": slr.models.CTCBeamSearchDecoder,
    "TFCTCBeamSearchDecoder": slr.models.TFCTCBeamSearchDecoder,
}

EvaluatorDict = {
    "sclite": slr.evaluation.ScliteEvaluator,
    # "python": None,
}

TransformDict = {
    "video": {
        'train': Compose([ToTensor(), RandomCrop(224), RandomHorizontalFlip(0.5), TemporalRescale(0.2),
                          Normalize(mean=[127.5, 127.5, 127.5], std=[127.5, 127.5, 127.5])]),
        'dev': Compose([ToTensor(), CenterCrop(224), Normalize(mean=[127.5, 127.5, 127.5], std=[127.5, 127.5, 127.5])]),
        'test': Compose([ToTensor(), CenterCrop(224), Normalize(mean=[127.5, 127.5, 127.5], std=[127.5, 127.5, 127.5])])
    },
    "keypoint": {
        'train': None,
        'dev': None,
        'test': None
    }
}
