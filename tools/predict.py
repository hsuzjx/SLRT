import lightning as L
import numpy as np

import slr.models
from slr.datasets import Phoenix2014DataModule

MODEL_NAME = 'CorrNet'
MODEL_PATH = '../experiments/Phoenix2014/CorrNet/2024-09-10_20-19-23/checkpoints/epoch=31-DEV_WER=19.50.ckpt'


def predict():
    with open('../data/global_files/gloss_dict/phoenix2014_gloss_dict.npy', 'rb') as f:
        gloss_dict = np.load(f, allow_pickle=True).item()
    datamodule = Phoenix2014DataModule(
        features_path='../data/phoenix2014/phoenix-2014-multisigner/features/fullFrame-256x256px',
        annotations_path='../data/phoenix2014/phoenix-2014-multisigner/annotations/manual',
        gloss_dict=gloss_dict,
        # ground_truth_path='../data/global_files/ground_truth',
        num_workers=8,
        batch_size=2
    )
    datamodule.setup(stage='fit')
    datamodule.setup(stage='test')
    dls = [datamodule.train_dataloader(), datamodule.val_dataloader(), datamodule.test_dataloader()]

    model = getattr(slr.model, MODEL_NAME).load_from_checkpoint(MODEL_PATH)
    trainer = L.Trainer(devices=[0])
    model.eval()

    res = trainer.predict(model, dataloaders=dls[2])

    res_sentence = []
    for batch in res:
        for sample in batch:
            res_sentence.append(" ".join([i[0] for i in sample]))

    return res_sentence


if __name__ == "__main__":
    s = predict()
    print(s)
