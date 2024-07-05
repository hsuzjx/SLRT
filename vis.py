import netron
import lightning as L
import numpy as np
import torch
from torch._C._onnx import TrainingMode

from model_interface import SLRModel

if __name__ == '__main__':
    model_path = '/new_home/xzj23/openmmlab_workspace/SLR/checkpoints/Phoenix2014_TTT-bh_2024-07-01_19:26:19/last.ckpt'
    onnx_path = '/new_home/xzj23/openmmlab_workspace/SLR/checkpoints/Phoenix2014_TTT-bh_2024-07-01_19:26:19/last.onnx'
    model = SLRModel(
        num_classes=1296, conv_type=2, use_bn=False, hidden_size=1024, gloss_dict_path='./.tmp',
        save_path='./.tmp/{}_{}'.format('test', 'test'),
        sh_path='./evaluation/slr_eval',
        ground_truth_path='./.tmp',
        mer_path='./evaluation/slr_eval',
        weight_norm=True,
        lr=0.0001, weight_decay=0.0001, lr_scheduler_milestones=None, lr_scheduler_gamma=0.2,
        last_epoch=-1,
        test_param=False,
    )

    # model = torch.load(model_path)
    model = model.load_from_checkpoint(model_path)
    l = dir(model)
    print('state_dict' in l)

    device = torch.device('cuda:1')
    x = torch.rand(1, 244, 3, 224, 224)
    lgt = torch.LongTensor([244])
    res = model(x.to(device), lgt.to(device))
    print(res)
    input_sample = (x.to(device), lgt.to(device))
    # model.to_onnx(onnx_path, input_sample=input_sample, export_params=True)
    torch.onnx.export(model, input_sample, onnx_path, export_params=True, )
    # netron.start(model_path, address=('10.12.44.154', 19876))
