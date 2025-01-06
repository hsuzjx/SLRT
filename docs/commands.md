# COMMANDS



## Train Commands

### XModel

```bash
MODEL=XModel
DATASET=Phoenix2014
DATA_TYPE=video
PROJECT=${DATASET}_Experiment
NAME=${MODEL}/xxx

python main.py project=${PROJECT} name=${NAME} model=${MODEL} dataset=${DATASET} dataloader=${DATA_TYPE} trainer=${DATA_TYPE} data_type=${DATA_TYPE} trainer.devices=[0] times=0
```

### MSKA

```bash
MODEL=MSKA
DATASET=Phoenix2014
DATA_TYPE=keypoint
PROJECT=${DATASET}_Experiment
NAME=${MODEL}/xxx

python main.py project=${PROJECT} name=${NAME} model=${MODEL} dataset=${DATASET} dataloader=${DATA_TYPE} trainer=${DATA_TYPE} data_type=${DATA_TYPE} trainer.devices=[0] times=0
```

