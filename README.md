# SLRT:  Toolbox for Sign Language Recognition and Translation based on PyTorch



## Introduction

TODO ...





## train

```
# MSKA patch-kps
python main.py project=Phoenix2014_Experiment name=MSKA_patchs-kps times=0 model=MSKA dataset=phoenix2014 dataloader=patch-kps trainer=patch-kps data_type=patch-kps dataloader.batch_size=4 trainer.devices=[0] num_threads=12 dataset.data_cfgs.features_dir=null dataloader.patch_hw=[5,5]
```

