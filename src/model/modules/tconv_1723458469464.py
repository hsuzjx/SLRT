import torch
import torch.nn as nn


class MultiScale_TemporalConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilations=None, ):
        super().__init__()
        if dilations is None:
            dilations = [1, 2, 3, 4]
        self.num_branches = len(dilations)

        # Temporal Convolution branches
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(
                    in_channels,
                    out_channels // self.num_branches,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    padding=dilation),
                nn.BatchNorm1d(out_channels // self.num_branches)
            )
            for dilation in dilations
        ])

    def forward(self, x):
        # Input dim: (N,C,T)
        branch_outs = []
        for tempconv in self.branches:
            out = tempconv(x)
            branch_outs.append(out)

        out = torch.cat(branch_outs, dim=1)
        return out


class TemporalConv(nn.Module):
    def __init__(self, input_size, hidden_size, conv_type=2, use_bn=False, num_classes=-1):
        super(TemporalConv, self).__init__()
        self.use_bn = use_bn
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        assert 0 <= conv_type <= 8, "Invalid conv_type"
        self.conv_type = conv_type

        # Simplified kernel size definition
        self.kernel_size = [
            ['K3'],
            ['K5', "P2"],
            ['K5', "P2", 'K5', "P2"],
            ['K5', 'K5', "P2"],
            ['K5', 'K5'],
            ['K5', "P2", 'K5'],
            ["P2", 'K5', 'K5'],
            ["P2", 'K5', "P2", 'K5'],
            ["P2", "P2", 'K5', 'K5']
        ][conv_type]

        modules = []
        for layer_idx, ks in enumerate(self.kernel_size):
            input_sz = self.input_size if layer_idx == 0 else self.hidden_size
            if ks[0] == 'P':
                modules.append(nn.MaxPool1d(kernel_size=int(ks[1]), ceil_mode=False))
            elif ks[0] == 'K':
                modules.append(nn.Conv1d(input_sz, self.hidden_size, kernel_size=int(ks[1]), stride=1, padding=0))
                if self.use_bn:
                    modules.append(nn.BatchNorm1d(self.hidden_size))
                modules.append(nn.ReLU(inplace=True))
        self.temporal_conv = nn.Sequential(*modules)

        if self.num_classes != -1:
            self.fc = nn.Linear(self.hidden_size, self.num_classes)

    def update_lgt(self, lgt):
        feat_len = lgt.clone()
        for ks in self.kernel_size:
            if ks[0] == 'P':
                feat_len = torch.ceil(feat_len / 2)
            else:
                feat_len -= int(ks[1]) - 1
        return feat_len

    def forward(self, frame_feat, lgt):
        visual_feat = self.temporal_conv(frame_feat)
        lgt = self.update_lgt(lgt)
        logits = None if self.num_classes == -1 else self.fc(visual_feat.transpose(1, 2)).transpose(1, 2)
        return {
            "visual_feat": visual_feat.permute(2, 0, 1),
            "conv_logits": logits.permute(2, 0, 1),
            "feat_len": lgt.cpu(),
        }
