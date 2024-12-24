import torch
import torch.nn as nn


class MultiScale_TemporalConv(nn.Module):
    """
    Multi-scale temporal convolution module, used to extract features from time series data at multiple scales.
    
    Attributes:
        in_channels (int): Number of channels in the input.
        out_channels (int): Number of channels in the output.
        kernel_size (int): Size of the convolution kernel.
        dilations (list): List of dilation rates for temporal convolutions.
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, dilations=[1, 2, 3, 4]):
        super().__init__()

        # Multiple branches of temporal convolution
        self.num_branches = 4

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

        # self.fuse = nn.Conv1d(in_channels * self.num_branches, out_channels, kernel_size=1)
        # self.fuse = nn.Conv2d(in_channels, out_channels, kernel_size=(4,1))
        # self.bn = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        """
        Forward propagation of the multi-scale temporal convolution module.
        
        Parameters:
            x (Tensor): Input tensor, dimension (N, C, T, V).
            
        Returns:
            Tensor: Output tensor after multi-scale temporal convolution, dimension (N, C_out, T, V).
        """
        branch_outs = []
        for tempconv in self.branches:
            out = tempconv(x)
            branch_outs.append(out)

        out = torch.cat(branch_outs, dim=1)
        # out = torch.stack(branch_outs, dim=2)
        # out = self.fuse(out).squeeze(2)
        # out = self.bn(out)
        return out


class TemporalConv(nn.Module):
    """
    Temporal convolution module, used for feature extraction and transformation in time series data.
    
    Attributes:
        input_size (int): Number of channels in the input.
        hidden_size (int): Number of channels in the hidden layer.
        conv_type (int): Type of convolution structure.
        use_bn (bool): Whether to use batch normalization.
        num_classes (int): Number of categories for classification, -1 if no classification is performed.
    """
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
            input_sz = self.input_size if layer_idx == 0 or self.conv_type == 6 and layer_idx == 1 or self.conv_type == 7 and layer_idx == 1 or self.conv_type == 8 and layer_idx == 2 else self.hidden_size
            if ks[0] == 'P':
                modules.append(nn.AvgPool1d(kernel_size=int(ks[1]), ceil_mode=False))
            elif ks[0] == 'K':
                modules.append(
                    nn.Conv1d(input_sz, self.hidden_size, kernel_size=int(ks[1]), stride=1, padding=0)
                    # MultiScale_TemporalConv(input_sz, self.hidden_size)
                )
                modules.append(nn.BatchNorm1d(self.hidden_size))
                modules.append(nn.ReLU(inplace=True))
        self.temporal_conv = nn.Sequential(*modules)

        if self.num_classes != -1:
            self.fc = nn.Linear(self.hidden_size, self.num_classes)

    def update_lgt(self, lgt):
        """
        Update the length of the feature based on the convolution structure.
        
        Parameters:
            lgt (Tensor): Original feature length.
            
        Returns:
            Tensor: Updated feature length.
        """
        feat_len = lgt.clone()
        for ks in self.kernel_size:
            if ks[0] == 'P':
                feat_len = torch.ceil(feat_len / 2).long()  # Ensure integer division
            else:
                feat_len -= int(ks[1]) - 1
                # pass
        return feat_len

    def forward(self, frame_feat, lgt):
        """
        Forward propagation of the temporal convolution module.
        
        Parameters:
            frame_feat (Tensor): Input frame feature, dimension (N, C_in, T, V).
            lgt (Tensor): Length of the input feature.
            
        Returns:
            dict: Containing visual features, convolution outputs, and updated feature lengths.
        """
        visual_feat = self.temporal_conv(frame_feat)
        lgt = self.update_lgt(lgt)
        logits = None if self.num_classes == -1 else self.fc(visual_feat.transpose(1, 2)).transpose(1, 2)
        return {
            "visual_feat": visual_feat.permute(2, 0, 1),
            "conv_logits": logits.permute(2, 0, 1),
            "feat_len": lgt.cpu(),
        }
