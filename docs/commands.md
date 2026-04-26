# COMMANDS



## Train Commands

### XModel

```bash
MODEL=XModel
DATASET=Phoenix2014
DATA_TYPE=video
PROJECT=${DATASET}_Experiment
NAME=${MODEL}/xxx
AFFINITY_CORE="0-9"

taskset -c ${AFFINITY_CORE} python main.py project=${PROJECT} name=${NAME} model=${MODEL} dataset=${DATASET} dataloader=${DATA_TYPE} trainer=${DATA_TYPE} data_type=${DATA_TYPE} trainer.devices=[0] times=0
```

### PatchModel

```bash
MODEL=PatchModel
DATASET=Phoenix2014
DATA_TYPE=patch-kps
PROJECT=${DATASET}_Experiment
NAME=${MODEL}/xxx
AFFINITY_CORE="0-9"

taskset -c ${AFFINITY_CORE} python main.py project=${PROJECT} name=${NAME} model=${MODEL} dataset=${DATASET} dataloader=${DATA_TYPE} trainer=${DATA_TYPE} data_type=${DATA_TYPE} dataset.data_cfgs.features_dir=null trainer.devices=[0] times=0
```

### KpsModel

```bash
MODEL=KpsModel
DATASET=Phoenix2014
DATA_TYPE=keypoint
PROJECT=${DATASET}_Experiment
NAME=${MODEL}/xxx
AFFINITY_CORE="0-9"

taskset -c ${AFFINITY_CORE} python main.py project=${PROJECT} name=${NAME} model=${MODEL} dataset=${DATASET} dataloader=${DATA_TYPE} trainer=${DATA_TYPE} data_type=${DATA_TYPE} trainer.devices=[0] times=0
```

