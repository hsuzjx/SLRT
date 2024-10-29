from torchvision.transforms import Compose, RandomCrop, RandomHorizontalFlip, Normalize, CenterCrop

import slr.datasets
import slr.models
from slr.datasets.transforms import ToTensor, TemporalRescale

DataModuleClassDict = {
    "phoenix2014": slr.datasets.Phoenix2014DataModule,
    "phoenix2014T": slr.datasets.Phoenix2014TDataModule,
    "csl-daily": slr.datasets.CSLDailyDataModule
}

ModelClassDict = {
    "CorrNet": slr.models.CorrNet,
    # "SLRTransformer": slr.models.SLRTransformer,
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
