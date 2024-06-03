import copy
import logging
import math
import torch
import torch.nn as nn
import torch.nn.functional as nnf
from torch.nn import Dropout, Softmax, Linear, Conv3d, LayerNorm
import sys
sys.path.append(".")
from models import Model#, Classifier
from cross_attention import CrossAttention



class Teacher(nn.Module):
    def __init__(self, config, img_size=256):
        super(Teacher, self).__init__()
        self.transformerX = Model(config, img_size)
        # self.decoder = DecoderCup(config, img_size)
        # self.reg_head = RegistrationHead(
        #     in_channels=config.decoder_channelsfg[-1],
        #     out_channels=config['n_dims'],
        #     kernel_size=3,
        # )
        # self.spatial_trans = SpatialTransformer(img_size)
        # self.config = config
        #self.integrate = VecInt(img_size, int_steps)
        '''self.transformerY = Transformer(config, img_size, vis)
        self.cross_att = CrossAttention(config.hidden_size, config.transformer.num_heads)'''
        #self.clf = Classifier()

        self.norm1 = nn.LayerNorm(config.hidden_size)
        self.norm2 = nn.LayerNorm(config.hidden_size)
    def forward(self, x):

        #source = x[:,0:1,:,:] #0: fixed image, 1: moving image

        x = self.transformerX(x)  # (B, n_patch, hidden) 
        #x = x.permute(1,0,2) #(based on sequence_length, bs, dims)
        # x = self.decoder(x, features)
        # flow = self.reg_head(x)
        # #flow = self.integrate(flow)
        # out = self.spatial_trans(source, flow)
        '''y, attn_weights_y, features_y = self.transformerY(y)
        #y = y.permute(1, 0, 2)
        cross_weightXY = self.cross_att(x.permute(1,0,2), y.permute(1,0,2))
        cross_weightXY = cross_weightXY.permute(1, 0, 2)
        cross_weightXY = self.norm1(cross_weightXY) #(bs, npatches, dims)

        cross_weightYX = self.cross_att(y.permute(1,0,2), x.permute(1,0,2))
        cross_weightYX = cross_weightYX.permute(1, 0, 2)
        cross_weightYX = self.norm2(cross_weightYX)

        cross_weight = torch.concat([cross_weightXY, cross_weightYX], -1) #-------------------------------------------------------------'''
        #feat = torch.concat([cross_weight, x, y], -1)
        # out =  self.clf(x)
        x = self.norm1(x)
        return x 

# if __name__=='__main__':
#     x= torch.rand(4, 1, 256, 256, 256).cuda() #(bs, c, t, h, w)
#     y= torch.rand(4, 1, 256, 256, 256).cuda()
#     config = get_3DReg_config()
#     model = ViTVNet(config, (256,256,256)).cuda()
#     out = model(x, y)
#     print(out.shape)