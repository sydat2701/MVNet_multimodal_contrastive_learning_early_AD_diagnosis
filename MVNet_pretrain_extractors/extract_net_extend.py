import torch
import torch.nn as nn
from components.resbased_block import generate_model
from components.depthwise_sep_conv import depthwise_separable_conv
from components.cbam import CBAM_Block
from components.resbase_extend_block import get_res_extend_block
import torch.nn.functional as F

def conv3x3x3(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes,
                     out_planes,
                     kernel_size=3,
                     stride=stride,
                     padding=1,
                     bias=False)


def conv1x1x1(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes,
                     out_planes,
                     kernel_size=1,
                     stride=stride,
                     bias=False)
class DownScale(nn.Module):

    def __init__(self, in_planes, planes, stride=1, downsample=False):
        super().__init__()

        self.conv1 = conv1x1x1(in_planes, planes)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = conv3x3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = conv1x1x1(planes, planes)
        self.bn3 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out = self.relu(out)

        return out

from teacher.NL_block import NONLocalBlock3D
from cross_attention import CrossAttention

class MVBlock(nn.Module):
    def __init__(self):
        super(MVBlock, self).__init__()
        self.conv2d_1x = depthwise_separable_conv(64, 128)
        self.conv2d_2x = depthwise_separable_conv(128, 256)
        self.conv2dx = nn.Sequential(self.conv2d_1x, nn.InstanceNorm2d(128), nn.LeakyReLU(0.2) , \
                                    self.conv2d_2x, nn.InstanceNorm2d(256), nn.LeakyReLU(0.2))
        
        self.resblock_1x, self.resblock_2x = get_res_extend_block()
        self.resblock_x = nn.Sequential(self.resblock_1x, self.resblock_2x)

    def forward(self, x):
        bs, c, h, t, w = x.shape
        cor_x = x.permute(0, 3, 1, 2, 4).contiguous().view(bs*t, c, h, w) 
        sag_x = x.permute(0, 4, 1, 2, 3).contiguous().view(bs*w, c, h, t)
        axl_x = x.permute(0, 2, 1, 3, 4).contiguous().view(bs*h, c, t, w)

        cor_x = self.conv2dx(cor_x)
        cor_x = cor_x.view(bs, cor_x.size(1), -1, cor_x.size(2), cor_x.size(3)) #(bs, c, t, h, w)
        sag_x = self.conv2dx(sag_x)
        sag_x = sag_x.view(bs, sag_x.size(1), -1, sag_x.size(2), sag_x.size(3))
        axl_x = self.conv2dx(axl_x)
        axl_x = axl_x.view(bs, axl_x.size(1), -1, axl_x.size(2), axl_x.size(3))

        res_featx = self.resblock_x(x)
        featx = torch.cat((cor_x, sag_x, axl_x, res_featx), dim=1)
        
        return featx


class ExtractNet(nn.Module):
    def __init__(self):
        super(ExtractNet, self).__init__()
        self.res_base = generate_model(10)
                
        self.attx = NONLocalBlock3D(in_channels=256*3+256, bn_layer=True)

        self.mvx = MVBlock()

        self.pool = nn.AdaptiveAvgPool3d(8)


    def forward(self, x):
        
        bs = x.shape[0]

        x = self.res_base(x) #shape: (bs,  64, 45, 45, 45)

        featx = self.mvx(x)     
        featx = self.pool(featx)
        featx, att_weight_imgx, att_value_imgx = self.attx(featx)
        featx = torch.cat([featx, self.pool(x)], dim=1)  

        return featx


class MVNet(nn.Module):
    def __init__(self):
        super(MVNet, self).__init__()
        self.extract_net = ExtractNet()

        self.fc1 = nn.Linear((256*3+256+64)*(8**3), 128)

    def forward(self, x):
        x = self.extract_net(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = F.normalize(x, dim=1)
        return x