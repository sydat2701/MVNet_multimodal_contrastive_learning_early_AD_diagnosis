import torch
import torch.nn as nn
from components.resbased_block import generate_model
from components.depthwise_sep_conv import depthwise_separable_conv
from components.cbam import CBAM_Block
from components.resbase_extend_block import get_res_extend_block
import torch.nn.functional as F
from components.resbased_block import BasicBlock

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
from q_pytorch_mri_tau_amyloid import AttentionModel



from contras_base_extract_net.net import get_contras_base_model
class ExtractNetConTras(nn.Module):
    def __init__(self, teacher_path, contras_weight_path, fol_idx):
        super(ExtractNetConTras, self).__init__()
        self.contras_basex = get_contras_base_model(contras_weight_path, fol_idx, 'MRI')
        self.contras_basey = get_contras_base_model(contras_weight_path, fol_idx, 'Amyloid')
        self.contras_basez = get_contras_base_model(contras_weight_path, fol_idx, 'Tau')
                
        #self.cbam = CBAM_Block(256*3+256)
        self.attx = NONLocalBlock3D(in_channels=1024, bn_layer=True)
        self.atty = NONLocalBlock3D(in_channels=1024, bn_layer=True)
        self.attz = NONLocalBlock3D(in_channels=1024, bn_layer=True)
        self.pool = nn.AdaptiveAvgPool3d(4)

        #downsample = self._downsample_basic_block()
        self.bottle_neckx = DownScale(2*1024, planes=8, stride=2)
        self.bottle_necky = DownScale(2*1024, planes=8, stride=2)
        self.bottle_neckz = DownScale(2*1024, planes=8, stride=2)

        self.f_extrax = BasicBlock(256*3+256, 256*3+256)
        self.f_extray = BasicBlock(256*3+256, 256*3+256)
        self.f_extraz = BasicBlock(256*3+256, 256*3+256)


        self.fc_fuse_att_surfx = nn.Linear(640*640, 128)
        self.fc_fuse_v_surfx = nn.Linear(640*192, 128)
        self.fc_att_weight_imgx = nn.Linear(512*128, 128)
        self.fc_att_value_imgx = nn.Linear(128*512, 128)

        self.fc_fuse_att_surfy = nn.Linear(640*640, 128)
        self.fc_fuse_v_surfy = nn.Linear(640*192, 128)
        self.fc_att_weight_imgy = nn.Linear(512*128, 128)
        self.fc_att_value_imgy = nn.Linear(128*512, 128)

        self.fc_fuse_att_surfz = nn.Linear(640*640, 128)
        self.fc_fuse_v_surfz = nn.Linear(640*192, 128)
        self.fc_att_weight_imgz = nn.Linear(512*128, 128)
        self.fc_att_value_imgz = nn.Linear(128*512, 128)
    

    def forward(self, x, y, z):
        
        bs = x.shape[0]

        featx_ori, x = self.contras_basex(x)
        featy_ori, y = self.contras_basey(y)
        featz_ori, z = self.contras_basez(z)

        featx = self.pool(self.f_extrax(featx_ori))
        featy = self.pool(self.f_extray(featy_ori))
        featz = self.pool(self.f_extraz(featz_ori))
       

        featx, att_weight_imgx, att_value_imgx = self.attx(featx)
        featx = torch.cat([featx, self.pool(featx_ori)], dim=1)
        featx = self.bottle_neckx(featx)

        featy, att_weight_imgy, att_value_imgy = self.atty(featy)
        featy = torch.cat([featy, self.pool(featy_ori)], dim=1)
        featy = self.bottle_necky(featy)

        featz, att_weight_imgz, att_value_imgz = self.attz(featz)
        featz = torch.cat([featz, self.pool(featz_ori)], dim=1)
        featz = self.bottle_neckz(featz)


        
        return featx, featy, featz


class SubClf(nn.Module):
    def __init__(self):
        super(SubClf, self).__init__()
        self.fc1 = nn.Linear(8*(2**3), 32)
        self.fc2 = nn.Linear(32, 1)
        self.sig = nn.Sigmoid()
        self.lkrelu = nn.LeakyReLU(0.2)
    def forward(self, x):
        x = self.fc1(x)
        x = self.lkrelu(x)
        x = self.fc2(x)
        x = self.sig(x)
        return x

class MVNetContras(nn.Module):
    def __init__(self, teacher_path, contras_weight_path, fol_idx):
        super(MVNetContras, self).__init__()
        self.extract_net = ExtractNetConTras(teacher_path, contras_weight_path, fol_idx)

        self.avg = nn.AdaptiveAvgPool3d(4)
        self.fc1 = nn.Linear(8*(2**3)*3, 32)
        self.lkrelu = nn.LeakyReLU(0.2)
        self.fc2 = nn.Linear(32, 1)
        self.sig = nn.Sigmoid()
        self.drop = nn.Dropout(0.4)
        self.drop1 = nn.Dropout(0.5)
        self.norm = nn.LayerNorm(8*(2**3)*3 ,eps=1e-6)
        self.sig1 = nn.Sigmoid()
        
        self.clf1 = SubClf()
        self.clf2 = SubClf()
        self.clf3 = SubClf()

    def forward(self, x, y, z):
        featx, featy, featz = self.extract_net(x, y, z)

        x = self.drop(featx)
        y = self.drop(featy)
        z = self.drop(featz)

        # latentx = self.avg(x)
        x_fl = x.view(x.size(0), -1)

        # latenty = self.avg(y)
        y_fl = y.view(y.size(0), -1)

        # latentz = self.avg(z)
        z_fl = z.view(z.size(0), -1)

        feat = torch.cat([x_fl, y_fl, z_fl], dim=1)
        feat = self.norm(feat)

        x = self.fc1(feat)
        x = x*self.sig1(x)
        x = self.lkrelu(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.sig(x)

        x_fl = self.clf1(x_fl)
        y_fl = self.clf2(y_fl)
        z_fl = self.clf3(z_fl)

        
        return x, x_fl, y_fl, z_fl