import numpy as np
import torch
from torch import nn
from tqdm import tqdm

from src.data import Phoenix2014Dataset
from src.data.transforms import *
from src.model.modules import SeqKD
from src2.model import SLRModel


def train(n_epoch=10):
    device = torch.device('npu:0')

    with open('/home/ma-user/work/workspace/SLR/data/global_files/gloss_dict/phoenix2014_gloss_dict.npy', 'rb') as f:
        gloss_dict = np.load(f, allow_pickle=True).item()

    dataset = Phoenix2014Dataset(
        features_path='/home/ma-user/work/workspace/SLR/data/phoenix2014/phoenix-2014-multisigner/features/fullFrame-256x256px',
        annotations_path='/home/ma-user/work/workspace/SLR/data/phoenix2014/phoenix-2014-multisigner/annotations/manual',
        gloss_dict=gloss_dict,
        mode='train',
        drop_ids=['13April_2011_Wednesday_tagesschau_default-14'],
        transform=Compose([RandomCrop(224), RandomHorizontalFlip(0.5), ToTensor(), TemporalRescale(0.2)])
    )

    dl = torch.utils.data.DataLoader(
        dataset,
        batch_size=2,
        num_workers=8,
        pin_memory=True,
        drop_last=True,
        collate_fn=dataset.collate_fn,
        shuffle=True
    )

    model = SLRModel().type(torch.npu.HalfTensor).to(device)

    loss_fn = dict()
    loss_fn["CTCLoss"] = nn.CTCLoss(reduction='none', zero_infinity=False).to(device)
    loss_fn['Distillation'] = SeqKD(T=8).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.0001)

    model.train()
    for epoch in range(n_epoch):
        for batch in tqdm(dl, desc=f'Epoch: {epoch}'):
            optimizer.zero_grad()

            x, x_lgt, y, y_lgt, info = batch
            x = x.type(torch.npu.HalfTensor).to(device)
            y = y.type(torch.npu.HalfTensor).to(device)
            # x = x.float().to(device)
            # y = y.float().to(device)
            x_lgt = x_lgt.to(device)
            y_lgt = y_lgt.to(device)

            conv1d_hat, y_hat, y_hat_lgt = model(x, x_lgt)
            conv1d_hat = conv1d_hat.type(torch.Tensor).to(device)
            y_hat = y_hat.type(torch.Tensor).to(device)
            conv1d_hat = conv1d_hat

            loss = 1.0 * loss_fn['CTCLoss'](conv1d_hat.log_softmax(-1), y,
                                            y_hat_lgt, y_lgt).mean() + \
                   1.0 * loss_fn['CTCLoss'](y_hat.log_softmax(-1), y,
                                            y_hat_lgt, y_lgt).mean() + \
                   25.0 * loss_fn['Distillation'](conv1d_hat, y_hat.detach(), use_blank=False)

            loss.backward()
            optimizer.step()


if __name__ == "__main__":
    train()
