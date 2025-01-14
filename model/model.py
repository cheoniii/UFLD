import torch
import torch.nn as nn

from model.backbone import resnet
import numpy as np

class conv_bn_relu(torch.nn.Module):
    def __init__(self,in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1,bias=False):
        super(conv_bn_relu,self).__init__()
        self.conv = torch.nn.Conv2d(in_channels,out_channels, kernel_size,
            stride = stride, padding = padding, dilation = dilation,bias = bias)
        self.bn = torch.nn.BatchNorm2d(out_channels)
        self.relu = torch.nn.ReLU()

    def forward(self,x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class parsingNet(nn.Module):
    def __init__(self, size=(288, 800), pretrained=True, backbone='50', cls_dim=(37, 10, 4), use_aux=False):
        """
        Initialize the parsingNet model.
        Args:
            size (tuple): Input image size (height, width).
            pretrained (bool): Whether to use pretrained backbone weights.
            backbone (str): Type of ResNet backbone ('18', '34', '50', etc.).
            cls_dim (tuple): Classification output dimensions (gridding, rows, lanes).
            use_aux (bool): Whether to use auxiliary headers for intermediate features.
        """
        super(parsingNet, self).__init__()

        self.size = size
        self.height, self.width = size
        self.cls_dim = cls_dim
        self.use_aux = use_aux
        self.total_dim = np.prod(cls_dim)

        # Backbone network
        self.model = resnet(backbone, pretrained=pretrained)

        # Auxiliary headers (only initialized if use_aux=True)
        if self.use_aux:
            self.aux_header2 = self._make_aux_header(128 if backbone in ['34', '18'] else 512)
            self.aux_header3 = self._make_aux_header(256 if backbone in ['34', '18'] else 1024)
            self.aux_header4 = self._make_aux_header(512 if backbone in ['34', '18'] else 2048)
            self.aux_combine = self._make_aux_combine_header(cls_dim[-1])

            # Initialize weights for auxiliary headers
            initialize_weights(self.aux_header2, self.aux_header3, self.aux_header4, self.aux_combine)

        # Pooling and classification headers
        self.pool = nn.Conv2d(512 if backbone in ['34', '18'] else 2048, 8, 1)

        # Calculate cls_in dynamically
        self.cls_in = (self.height // 32) * (self.width // 32) * 8

        # Classification headers
        self.cls = nn.Sequential(
            nn.Linear(self.cls_in, 2048),
            nn.ReLU(),
            nn.Linear(2048, self.total_dim)
        )

        # Initialize classification weights
        initialize_weights(self.cls)

    def _make_aux_header(self, in_channels):
        """
        Helper function to create auxiliary headers.
        """
        return nn.Sequential(
            conv_bn_relu(in_channels, 128, kernel_size=3, stride=1, padding=1),
            conv_bn_relu(128, 128, kernel_size=3, padding=1),
            conv_bn_relu(128, 128, kernel_size=3, padding=1),
        )

    def _make_aux_combine_header(self, out_channels):
        """
        Helper function to create auxiliary combine header.
        """
        return nn.Sequential(
            conv_bn_relu(384, 256, 3, padding=2, dilation=2),
            conv_bn_relu(256, 128, 3, padding=2, dilation=2),
            conv_bn_relu(128, 128, 3, padding=2, dilation=2),
            conv_bn_relu(128, 128, 3, padding=4, dilation=4),
            nn.Conv2d(128, out_channels + 1, 1)
        )

    def _calculate_cls_in(self):
        """
        Calculate the input size for the classification layer based on the input dimensions.
        input_size: (288, 400) -> cls_in: 936
        input_size: (288, 800) -> cls_in: 1800
        input_size: (144, 400) -> cls_in: 520
        input_size: (144, 384) -> cls_in: 480
        """
        return (self.height // 32) * (self.width // 32) * 8

    def forward(self, x):
        """
        Forward pass for parsingNet.
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width).
        Returns:
            group_cls: Output classification results.
            aux_seg (optional): Auxiliary segmentation map.
        """
        batch_size, _, height, width = x.shape
        x2, x3, fea = self.model(x)

        if self.use_aux:
            x2 = self.aux_header2(x2)
            x3 = self.aux_header3(x3)
            x4 = self.aux_header4(fea)

            x3 = torch.nn.functional.interpolate(x3, size=(x2.shape[2], x2.shape[3]), mode='bilinear', align_corners=False)
            x4 = torch.nn.functional.interpolate(x4, size=(x2.shape[2], x2.shape[3]), mode='bilinear', align_corners=False)

            aux_seg = torch.cat([x2, x3, x4], dim=1)
            aux_seg = self.aux_combine(aux_seg)
        else:
            aux_seg = None

        fea = self.pool(fea).view(batch_size, -1)
        group_cls = self.cls(fea).view(batch_size, *self.cls_dim)

        if self.use_aux:
            return group_cls, aux_seg

        return group_cls



def initialize_weights(*models):
    for model in models:
        real_init_weights(model)
def real_init_weights(m):

    if isinstance(m, list):
        for mini_m in m:
            real_init_weights(mini_m)
    else:
        if isinstance(m, torch.nn.Conv2d):
            torch.nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)
        elif isinstance(m, torch.nn.Linear):
            m.weight.data.normal_(0.0, std=0.01)
        elif isinstance(m, torch.nn.BatchNorm2d):
            torch.nn.init.constant_(m.weight, 1)
            torch.nn.init.constant_(m.bias, 0)
        elif isinstance(m,torch.nn.Module):
            for mini_m in m.children():
                real_init_weights(mini_m)
        else:
            print('unkonwn module', m)