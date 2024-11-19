import torch
from torchvision.transforms import Compose, RandomCrop, Normalize, RandomHorizontalFlip, CenterCrop

import slr.datasets
import slr.models
from slr.datasets.transforms import ToTensor, TemporalRescale

CONFIG_PATH = '../configs'
CONFIG_NAME = 'CorrNet_CSL-Daily_experiment.yaml'

# TokenizerDict = {
#
# }

DataModuleClassDict = {
    "phoenix2014": slr.datasets.Phoenix2014DataModule,
    "phoenix2014T": slr.datasets.Phoenix2014TDataModule,
    "csl-daily": slr.datasets.CSLDailyDataModule,
    "phoenix2014-keypoint": slr.datasets.Phoenix2014KeypointDataModule,
    "phoenix2014T-keypoint": slr.datasets.Phoenix2014TKeypointDataModule,
    "csl-daily-keypoint": slr.datasets.CSLDailyKeypointDataModule,
}

ModelClassDict = {
    "CorrNet": slr.models.CorrNet,
    # "SLRTransformer": slr.models.SLRTransformer,
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

transform = {
    # 'train': Compose([ToTensor(), RandomCrop(224),]),
    'train': Compose([ToTensor(), RandomCrop(224), RandomHorizontalFlip(0.5), TemporalRescale(0.2),
                      Normalize(mean=[127.5, 127.5, 127.5], std=[127.5, 127.5, 127.5])]),
    'dev': Compose([ToTensor(), CenterCrop(224), Normalize(mean=[127.5, 127.5, 127.5], std=[127.5, 127.5, 127.5])]),
    'test': Compose([ToTensor(), CenterCrop(224), Normalize(mean=[127.5, 127.5, 127.5], std=[127.5, 127.5, 127.5])])
}
# transform = {
#     'train': Compose([RandomCrop(224), RandomHorizontalFlip(0.5), TemporalRescale(0.2)]),
#     'dev': Compose([CenterCrop(224)]),
#     'test': Compose([CenterCrop(224)])
# }

# kps
# transform = {
#     'train': None,
#     'dev': None,
#     'test': None
# }
