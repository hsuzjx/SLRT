import torch
import transformers
from torchvision.transforms import Compose, RandomCrop, Normalize, RandomHorizontalFlip, CenterCrop

import slrt
from slrt.datasets.transforms import ToTensor, TemporalRescale, RandomDrop
from slrt.datasets.transforms.keypoints import SelectIndex, RandomMove

CONFIG_PATH = '../configs'
CONFIG_NAME = 'config.yaml'

DataModuleClassDict = {
    "video": {
        "Phoenix2014": slrt.datasets.DataModules.Phoenix2014DataModule,
        "Phoenix2014T": slrt.datasets.DataModules.Phoenix2014TDataModule,
        "CSL-Daily": slrt.datasets.DataModules.CSLDailyDataModule,
    },
    "keypoint": {
        "Phoenix2014": slrt.datasets.DataModules.Phoenix2014KeypointDataModule,
        "Phoenix2014T": slrt.datasets.DataModules.Phoenix2014TKeypointDataModule,
        "CSL-Daily": slrt.datasets.DataModules.CSLDailyKeypointDataModule,
    },
    "patch-kps": {
        "Phoenix2014": slrt.datasets.DataModules.Phoenix2014PatchKpsDataModule,
        "Phoenix2014T": slrt.datasets.DataModules.Phoenix2014TPatchKpsDataModule,
        "CSL-Daily": slrt.datasets.DataModules.CSLDailyPatchKpsDataModule,
    }
}

ModelClassDict = {
    # "XModel": slrt.models.XModel,
    # "PatchModel": slrt.models.PatchModel,
    "KpsModel": slrt.models.KpsModel,
}

InputSampleDict = {
    "XModel": (torch.randn(1, 100, 3, 224, 224).to('cpu'), torch.LongTensor([100]).to('cpu')),
    "PatchModel": (torch.randn(1, 100, 3, 79, 13, 13).to('cpu'), torch.LongTensor([100]).to('cpu'),
                   torch.randn(1, 100, 79, 3).to('cpu')),
    "KpsModel": (torch.randn(1, 3, 100, 133).to('cpu'), torch.LongTensor([100]).to('cpu')),
}

TokenizerDict = {
    "Recognition": {
        "SimpleTokenizer": slrt.datasets.Tokenizers.SimpleTokenizer,
    },
    "Translation": {
        "BertTokenizer": transformers.BertTokenizer,
    }
}

DecoderDict = {
    "Recognition": {
        "CTCBeamSearchDecoder": slrt.models.CTCBeamSearchDecoder,
        # "TFCTCBeamSearchDecoder": slrt.models.TFCTCBeamSearchDecoder,
    },
    "Translation": {
        "CTCBeamSearchDecoder": slrt.models.CTCBeamSearchDecoder,
        # "TFCTCBeamSearchDecoder": slrt.models.TFCTCBeamSearchDecoder,
    }
}

EvaluatorDict = {
    "Recognition": {
        "sclite": slrt.evaluation.ScliteEvaluator,
        "python": slrt.evaluation.PythonEvaluator,
    },
    "Translation": {
        "python": slrt.evaluation.translation.PythonTranslationEvaluator,
    }
}

TransformDict = {
    "video": {
        'train': Compose([
            ToTensor(), RandomCrop(224), RandomHorizontalFlip(0.5), TemporalRescale(0.2),
            Normalize(mean=[127.5, 127.5, 127.5], std=[127.5, 127.5, 127.5])
        ]),
        'dev': Compose([
            ToTensor(), CenterCrop(224),
            Normalize(mean=[127.5, 127.5, 127.5], std=[127.5, 127.5, 127.5])
        ]),
        'test': Compose([
            ToTensor(), CenterCrop(224),
            Normalize(mean=[127.5, 127.5, 127.5], std=[127.5, 127.5, 127.5])
        ])
    },
    "keypoint": {
        'train': Compose([
            SelectIndex(0.5, 1.5, 400),
            # RandomDrop(0.3),
            RandomMove()
        ]),
        'dev': Compose([
            SelectIndex(1, 1, 400)
        ]),
        'test': Compose([
            SelectIndex(1, 1, 400)
        ])
    },
    "patch-kps": {
        'train': slrt.datasets.transforms.patchkps.Compose([
            slrt.datasets.transforms.patchkps.SelectIndex(0.5, 1.5, 400),
        ]),
        'dev': slrt.datasets.transforms.patchkps.Compose([
            slrt.datasets.transforms.patchkps.SelectIndex(1, 1, 400),
        ]),
        'test': slrt.datasets.transforms.patchkps.Compose([
            slrt.datasets.transforms.patchkps.SelectIndex(1, 1, 400),
        ]),
    }
}
