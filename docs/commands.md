# COMMANDS



## Train Commands

### XModel

```bash
MODEL=XModel
DATASET=phoenix2014
DATA_TYPE=video
PROJECT=Phoenix2014_Experiment
NAME=XModel_xxx

python main.py project=${PROJECT} name=${NAME} model=${MODEL} dataset=${DATASET} dataloader=${DATA_TYPE} trainer=${DATA_TYPE} data_type=${DATA_TYPE} trainer.devices=[0] times=0
```

### MSKA

```bash
MODEL=MSKA
DATASET=phoenix2014
DATA_TYPE=keypoint
PROJECT=Phoenix2014_Experiment
NAME=MSKA_xxx

python main.py project=${PROJECT} name=${NAME} model=${MODEL} dataset=${DATASET} dataloader=${DATA_TYPE} trainer=${DATA_TYPE} data_type=${DATA_TYPE} trainer.devices=[0] times=0
```

