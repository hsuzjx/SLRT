import torch

INPUT_SAMPLE = {
    "CorrNet": (torch.randn(2, 100, 3, 224, 224).to('cpu'), torch.LongTensor([100, 100]).to('cpu'))
}
