import torch
import torch.nn as nn

class CrossAttention(nn.Module):
    def __init__(self, input_dim, kernel_size=3):
        super(CrossAttention, self).__init__()

        self.conv_q = nn.Conv3d(input_dim, input_dim, kernel_size= kernel_size, padding = (kernel_size - 1) // 2)
        self.conv_k = nn.Conv3d(input_dim, input_dim, kernel_size= kernel_size, padding = (kernel_size - 1) // 2)
        self.conv_v = nn.Conv3d(input_dim, input_dim, kernel_size= kernel_size, padding = (kernel_size - 1) // 2)
        self.softmax = nn.Softmax(dim=-1)
        self.drop = nn.Dropout(0.2)
        self.norm = nn.LayerNorm(1024*(8**3))

    def forward(self, x, y, mask=None):
        q = self.conv_q(x)
        k = self.conv_k(y)
        v = self.conv_v(y)

        q = q.view(q.size(0), q.size(1), -1).transpose(1, 2)
        k = k.view(k.size(0), k.size(1), -1)
        v = v.view(v.size(0), v.size(1), -1)

        scores = torch.matmul(q, k)

        if mask is not None:
            scores.masked_fill_(mask == 0, float('-inf'))

        attention_weights = self.softmax(scores)
        attention_weights = self.drop(attention_weights)
        output = torch.matmul(attention_weights, v.transpose(-1, -2))
        output = output.permute(0, 2, 1).contiguous()


        output = output.view(output.size(0), output.size(1), *y.size()[2:])
        # 
        output = output + x
        '''tmp = output
        
        # print(output.shape)

        batch_size, channels, length, height, width = output.size()

        #output = output.permute(0, 2,3,4,1)
        output = output.view(batch_size, -1)
        output = self.norm(output)
        output = output.view(batch_size, channels, length, height, width)
        #output = output.permute(0, 4, 1,2,3)
        output = output + tmp'''

        return output