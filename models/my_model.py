import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
from functools import partial
from einops import rearrange
from einops.layers.torch import Rearrange
import random
__all__ = ['ResNet', 'resnet10', 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152', 'resnet200']


def conv3x3x3(in_planes, out_planes, stride=1):
    # 3x3x3 convolution with padding
    return nn.Conv3d(in_planes, out_planes, kernel_size=3,
                     stride=stride, padding=1, bias=False)


def downsample_basic_block(x, planes, stride):
    out = F.avg_pool3d(x, kernel_size=1, stride=stride)
    zero_pads = torch.Tensor(out.size(0), planes - out.size(1), out.size(2), out.size(3), out.size(4)).zero_()
    if isinstance(out.data, torch.cuda.FloatTensor):
        zero_pads = zero_pads.cuda()

    out = Variable(torch.cat([out.data, zero_pads], dim=1))

    return out
'''
backbone
'''

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x3(planes, planes)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = nn.Conv3d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = nn.Conv3d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class avgpool_choose(nn.Module):

    def __init__(self, opt):
        super(avgpool_choose, self).__init__()
        self.opt = opt
        self.avgpool_fmri = nn.AvgPool3d((math.ceil(opt.sample_duration_fmri / 16),
                                          math.ceil(opt.sample_size2_fmri/ 32),math.ceil(opt.sample_size1_fmri/ 32)), stride=1)
        self.avgpool_dti = nn.AvgPool3d((math.ceil(opt.sample_size1_dti/ 16),
                                          math.ceil(opt.sample_size2_dti/ 32),math.ceil(opt.sample_duration_dti/ 32)), stride=1)
        self.avgpool_dfc = nn.AvgPool3d((math.ceil(opt.sample_duration_dfc / 11),
                                          math.ceil(opt.sample_size1_fc/ 32),math.ceil(opt.sample_size2_fc/ 48)), stride=1)
        self.avgpool_zfc = nn.AvgPool3d((math.ceil(opt.sample_size2_fc/ 16),
                                          math.ceil(opt.sample_size1_fc/ 32), 1), stride=1)
        self.avgpool_dfc_half = nn.AvgPool3d((math.ceil(opt.sample_duration_dfc / 16),
                                          math.ceil(opt.sample_size1_fc/ 32),math.ceil(opt.sample_size2_fc/ 64)), stride=1)
        self.avgpool_dfc_quarter = nn.AvgPool3d((math.ceil(opt.sample_duration_dfc / 16),
                                          math.ceil(opt.sample_size1_fc/ 32),math.ceil(opt.sample_size2_fc/ 128)), stride=1)
        self.avgpool_zfc = nn.AvgPool3d((math.ceil(opt.sample_size2_fc/ 16),
                                          math.ceil(opt.sample_size1_fc/ 32), 1), stride=1)
    def forward(self, x):
        shape_res_H = x.shape[2]
        shape_res_W = x.shape[3]
        shape_res_T = x.shape[4]
        if shape_res_H == self.opt.sample_size1_fmri and shape_res_W == self.opt.sample_size2_fmri:
            avgpool = self.avgpool_fmri
        elif shape_res_H == self.opt.sample_size1_dti and shape_res_T == self.opt.sample_duration_dti:
            avgpool = self.avgpool_dti
        elif shape_res_H == self.opt.sample_size1_fc and shape_res_T == self.opt.sample_duration_zfc:
            avgpool = self.avgpool_zfc
        elif shape_res_H == self.opt.sample_size1_fc and shape_res_T == self.opt.sample_duration_dfc:
            avgpool = self.avgpool_dfc
        elif shape_res_H == self.opt.sample_size1_fc and shape_res_T == round(self.opt.sample_duration_dfc/2):
            avgpool = self.avgpool_dfc_half
        elif  shape_res_H == self.opt.sample_size1_fc and shape_res_T == round(self.opt.sample_duration_dfc/4):
            avgpool = self.avgpool_dfc_quarter
        else:
            avgpool = self.avgpool_fmri
        return avgpool
class ResNet(nn.Module):

    def __init__(self, block, layers, opt, shortcut_type='B', num_classes=400, t_stride=2):
        # self.last_fc = last_fc
        super(ResNet, self).__init__()
        self.opt = opt
        self.inplanes = 64
        self.batch_size = opt.batch_size
        super(ResNet, self).__init__()
        self.t_stride = t_stride
        self.conv1 = nn.Conv3d(1, 64, kernel_size=7, stride=(1, 2, self.t_stride),
                               padding=(3, 3, 3), bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], shortcut_type)
        self.layer2 = self._make_layer(block, 512, layers[1], shortcut_type, stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], shortcut_type, stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], shortcut_type, stride=2)
        self.pool_choose = avgpool_choose(opt)
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, shortcut_type, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            if shortcut_type == 'A':
                downsample = partial(downsample_basic_block,
                                     planes=planes * block.expansion,
                                     stride=stride)
            else:
                downsample = nn.Sequential(
                    nn.Conv3d(self.inplanes, planes * block.expansion,
                              kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm3d(planes * block.expansion)
                )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)
    def forward(self, x):
        avgpool = self.pool_choose(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = avgpool(x)
        return x
class dfc_encoder(nn.Module):
    def __init__(self, stride_1, stride_2, channel=8):
        super(dfc_encoder, self).__init__()
        self.conv1 = nn.Conv3d(1, 512, kernel_size=3, stride=(stride_1,stride_1, 2),
                               padding=(1, 1, 1), bias=False)
        self.bn1 = nn.BatchNorm3d(512)
        self.conv2 = nn.Conv3d(512, channel, kernel_size=3, stride=(stride_2, stride_2, 2),
                               padding=(1, 1, 1), bias=False)
        self.bn2 = nn.BatchNorm3d(channel)
        self.maxpool = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=1, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv_res = nn.Conv3d(1, channel, kernel_size=3, stride=(stride_1*stride_2, stride_1*stride_2, 4),
                               padding=(1, 1, 1), bias=False)
        self.bn_res = nn.BatchNorm3d(channel)
    def forward(self, x):
        x_res = self.conv_res(x)
        x_res = self.bn_res(x_res)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = x_res + x
        x = self.relu(x)
        x = self.maxpool(x)

        return x
class dfc_3d_downsample(nn.Module):
    def __init__(self, channel_in=8, channel_out=8, stride=2):
        super(dfc_3d_downsample , self).__init__()
        self.conv1 = nn.Conv3d(channel_in, 512, kernel_size=3, stride=(1, 1, 2),
                               padding=(1, 1, 1), bias=False)
        self.bn1 = nn.BatchNorm3d(512)
        self.conv2 = nn.Conv3d(512, channel_out, kernel_size=3, stride=(2, 2, stride),
                               padding=(1, 1, 1), bias=False)
        self.bn2 = nn.BatchNorm3d(channel_out)
        self.maxpool = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=1, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv_res = nn.Conv3d(channel_in, channel_out, kernel_size=3, stride=(2, 2, 4),
                               padding=(1, 1, 1), bias=False)
        self.bn_res = nn.BatchNorm3d(channel_out)
    def forward(self, x):
        x_res = self.conv_res(x)
        x_res = self.bn_res(x_res)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = x_res + x
        x = self.relu(x)
        x = self.maxpool(x)
        return x

# DD2C means depth_domain_2D_convolution 
class DD2C(nn.Module):
    def __init__(self):
            super(DD2C, self).__init__()
            self.conv1 = nn.Conv2d(in_channels=2, out_channels=512, kernel_size=1)
            self.bn1 = nn.BatchNorm2d(512)
            self.conv2 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=1)
            self.bn2 = nn.BatchNorm2d(512)
            self.relu = nn.ReLU(inplace=True)
            self.conv3 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=1)
            self.bn3 = nn.BatchNorm2d(512)
            self.conv4 = nn.Conv2d(in_channels=512, out_channels=1, kernel_size=1)
            self.bn4 = nn.BatchNorm2d(1)
            # self.conv5 = nn.Conv2d(in_channels=128, out_channels=1, kernel_size=1)
            # self.bn5 = nn.BatchNorm2d(1)
            self.conv_res = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=1)
            self.bn_res = nn.BatchNorm2d(1)
            self.relu = nn.ReLU(inplace=True)
            self.maxpool = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=1, padding=1)
    def forward(self, x):
        x_large = x[0]
        slice_tmp = x_large[:,0,:,:,0:1]
        channel_tmp = x_large[:,0:1,:,:,:]
        x_small = x[1]
        DD2C_factor = round(x_large.shape[4]/x_small.shape[4])
        for k in range(x_small.shape[1]):
            for i in range(0, (x_small.shape[4])+1):
                for j in range(1, DD2C_factor+1):
                  if j + i * DD2C_factor >= x_large.shape[4] or i == x_small.shape[4]:
                        break
                  slice_tensor_arr = torch.cat([torch.unsqueeze(x_large[:,k,:,:,j + i * DD2C_factor], dim=3),
                                                   torch.unsqueeze(x_small[:,k,:,:,i], dim=3)], dim=3)
                  slice_tensor_arr = torch.transpose(slice_tensor_arr, 1, 3)
                  slice_res = self.conv_res(slice_tensor_arr)
                  slice_res = self.bn_res(slice_res)
                  slice = self.conv1(slice_tensor_arr)
                  slice = self.bn1(slice)
                  slice = self.conv2(slice)
                  slice = self.bn2(slice)
                  slice = self.relu(slice)
                  slice = self.maxpool(slice)
                  slice = self.conv3(slice)
                  slice = self.bn3(slice)
                  slice = self.conv4(slice)
                  slice = self.bn4(slice)
                  slice = slice_res + slice
                  slice = self.relu(slice)
                  slice = self.maxpool(slice)
                  slice = torch.transpose(slice, 3, 1)
                  slice_tmp = torch.cat([slice_tmp, slice], dim=3)
            slice_tmp_channel = torch.unsqueeze(slice_tmp, dim=1)
            slice_tmp = x_large[:, 0, :, :, 0:1]
            slice_tmp_channel_T = slice_tmp_channel.shape[4]
            channel_tmp = torch.cat([slice_tmp_channel, channel_tmp[:,:,:,:,0:slice_tmp_channel_T]], dim=1)
        return channel_tmp[:,1:,:,:,:]


class MTSA(nn.Module):

    def __init__(self, block, layers, opt, shortcut_type='B', num_classes=400):
        super(dfc_pyramid, self).__init__()
        self.dfc_encoder = dfc_encoder(stride_1=2, stride_2=2, channel=8)
        self.dfc_encoder_5 = dfc_encoder(stride_1=2, stride_2=2, channel=8)
        self.dfc_encoder_10 = dfc_encoder(stride_1=2, stride_2=4, channel=16)
        self.dfc_encoder_20 = dfc_encoder(stride_1=4, stride_2=4, channel=32)
        self.dfc_encoder_40 = dfc_encoder(stride_1=4, stride_2=8, channel=64)
        self.dfc_3d_downsample = dfc_3d_downsample(channel_in=8, channel_out=8)
        self.dfc_3d_downsample_1_5 = dfc_3d_downsample(channel_in=8, channel_out=16)
        self.dfc_3d_downsample_1_5_10 = dfc_3d_downsample(channel_in=16, channel_out=32)
        self.dfc_3d_downsample_1_5_10_20 = dfc_3d_downsample(channel_in=32, channel_out=64)
        self.dfc_3d_downsample_1_5_10_20_40 = dfc_3d_downsample(channel_in=64, channel_out=128)
        self.DD2C_1_5 = DD2C()
        self.DD2C_1_5_10 = DD2C()
        self.DD2C_1_5_10_20 = DD2C()
        self.DD2C_1_5_10_20_40 = DD2C()
        self.DD2C = DD2C()
        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool3d((2,2,1), stride=1)
        self.conv_last_1 = nn.Conv3d(128, 128, kernel_size=3, stride=(1, 1, 1),
                               padding=(1, 1, 1), bias=False)
        self.bn_last_1 = nn.BatchNorm3d(128)
        self.conv_last_2 = nn.Conv3d(128, 128, kernel_size=3, stride=(1, 1, 1),
                               padding=(1, 1, 1), bias=False)
        self.bn_last_2 = nn.BatchNorm3d(128)
        self.conv_last_res = nn.Conv3d(128, 128, kernel_size=3, stride=(1, 1, 1),
                               padding=(1, 1, 1), bias=False)
        self.bn_last_res = nn.BatchNorm3d(128)
    def forward(self, x):
        shape_res_T = x.shape[4]
        # x_full = self.cnn_backbone(x)
        x_5 = x[:, :, :, :, 0:1]
        x_10 = x[:, :, :, :, 0:1]
        x_20 = x[:, :, :, :, 0:1]
        x_40 = x[:, :, :, :, 0:1]
        '''
        dfc build
        '''
        for i in range(0, shape_res_T, 5):
           if i+5 > shape_res_T:
              x_5 = torch.cat([x_5,torch.var_mean(x[:,:,:,:,i:],dim=4,keepdim=True)[1]], dim=4)
           else :
               x_5 = torch.cat([x_5, torch.var_mean(x[:, :, :, :, i:i+5], dim=4, keepdim=True)[1]], dim=4)
        for i in range(0,shape_res_T, 10):
           if i+10 > shape_res_T:
              x_10 = torch.cat([x_10,torch.var_mean(x[:,:,:,:,i:],dim=4,keepdim=True)[1]], dim=4)
           else :
               x_10 = (torch.cat([x_10, torch.var_mean(x[:, :, :, :, i:i+10], dim=4, keepdim=True)[1]], dim=4))
        for i in range(0,shape_res_T, 20):
           if i+20 > shape_res_T:
              x_20 = torch.cat([x_20,torch.var_mean(x[:,:,:,:,i:],dim=4,keepdim=True)[1]], dim=4)
           else :
               x_20 = (torch.cat([x_20, torch.var_mean(x[:, :, :, :, i:i+20], dim=4, keepdim=True)[1]], dim=4))
        for i in range(0,shape_res_T, 40):
           if i+40 > shape_res_T:
              x_40 = torch.cat([x_40,torch.var_mean(x[:,:,:,:,i:],dim=4,keepdim=True)[1]], dim=4)
           else :
               # x_80 = (torch.cat([x_80, torch.var_mean(x[:, :, :, :, i-80:i], dim=4, keepdim=True)[1]], dim=4))
               x_40 = (torch.cat([x_40, torch.var_mean(x[:, :, :, :, i:i+40], dim=4, keepdim=True)[1]], dim=4))
        '''
        MTSA step by step
        '''
        x_1 = self.dfc_encoder(x)
        x_5 = self.dfc_encoder_5(x_5[:,:,:,:,1:])
        x_1_5 = self.DD2C([x_1, x_5])
        x_1_5 = self.dfc_3d_downsample_1_5(x_1_5)
        x_10 = self.dfc_encoder_10(x_10[:,:,:,:,1:])
        x_1_5_10 = self.DD2C([x_1_5, x_10])
        x_1_5_10 = self.dfc_3d_downsample_1_5_10(x_1_5_10)
        x_20 = self.dfc_encoder_20(x_20[:,:,:,:,1:])
        x_1_5_10_20 = self.DD2C([x_1_5_10, x_20])
        x_1_5_10_20= self.dfc_3d_downsample_1_5_10_20(x_1_5_10_20)
        x_40 = self.dfc_encoder_40(x_40[:,:,:,:,1:])
        x_1_5_10_20_40 = self.DD2C([x_1_5_10_20, x_40])
        x_1_5_10_20_40 = self.dfc_3d_downsample_1_5_10_20_40(x_1_5_10_20_40)
        x_res = self.conv_last_res(x_1_5_10_20_40)
        x_res = self.bn_last_res(x_res)
        '''
        3d upsample
        '''
        x = self.conv_last_1(x_1_5_10_20_40)
        x = self.bn_last_1(x)
        x = self.conv_last_2(x)
        x = self.bn_last_2(x)
        x = self.relu(x)
        x = x + x_res
        x = self.avgpool(x)
        return x

class contrastive_loss(nn.Module):
    def __init__(self, opt, n_views=512):
        super(contrastive_loss, self).__init__()
        self.temperature=opt.temperature
        self.n_views = n_views
        self.criterion = torch.nn.CrossEntropyLoss().cuda()
    def info_nce_loss(self, features):
            batch_size = features[0].shape[0]
            labels = torch.cat([torch.arange(batch_size) for i in range(self.n_views)], dim=0)
            labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
            labels = labels.cuda()
            #        features = torch.cat(features, dim=1)
            features_1 = F.normalize(features[0], dim=1)
            features_2 = F.normalize(features[1], dim=1)
            similarity_matrix = torch.matmul(features_1.reshape(batch_size * self.n_views, 1),
                                             features_2.reshape(batch_size * self.n_views, 1).T)
            mask = torch.eye(labels.shape[0], dtype=torch.bool).cuda()
            labels = labels[~mask].view(labels.shape[0], -1)
            similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
            positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)
            negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)
            logits = torch.cat([positives, negatives], dim=1)
            labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()
            logits = logits / self.temperature
            return logits, labels

    def forward(self, features):
        logits, labels = self.info_nce_loss(features)
        cl_loss = self.criterion(logits, labels)
        return cl_loss
def pair(t):
    return t if isinstance(t, tuple) else (t, t)

def posemb_sincos_1d(patches, temperature = 10000, dtype = torch.float32):
    _, n, dim, device, dtype = *patches.shape, patches.device, patches.dtype

    n = torch.arange(n, device = device)
    assert (dim % 2) == 0, 'feature dimension must be multiple of 2 for sincos emb'
    omega = torch.arange(dim // 2, device = device) / (dim // 2 - 1)
    omega = 1. / (temperature ** omega)

    n = n.flatten()[:, None] * omega[None, :]
    pe = torch.cat((n.sin(), n.cos()), dim = 1)
    return pe.type(dtype)
class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )
    def forward(self, x):
        return self.net(x)
class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64):
        super().__init__()
        inner_dim = dim_head *  heads
        self.dim_head = dim_head
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim = -1)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        self.to_k = nn.Linear(dim, inner_dim, bias = False)
        self.to_v = nn.Linear(dim, inner_dim, bias = False)
        self.to_out = nn.Linear(inner_dim, dim, bias = False)

    def forward(self, x, cross=False):
        x = self.norm(x)
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)
        if cross == True:
            q = self.to_q(x[0])
            k = self.to_k(x[1])
            v = self.to_k(x[1])
        dots_qk = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        dots_qv = torch.matmul(q, v.transpose(-1, -2)) * self.scale
        dots_kv = torch.matmul(k, v.transpose(-1, -2)) * self.scale
        attn_qk = self.attend(dots_qk)
        attn_qv = self.attend(dots_qv)
        attn_kv = self.attend(dots_kv)
        out_qv = torch.matmul(attn_qv, k)
        out_qk = torch.matmul(attn_qk, v)
        out_kv = torch.matmul(attn_kv, q)
        out = out_qk
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)#dim=512, inner_dim=dim_head *  heads

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, mlp_dim, dim_head=64):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.dim = dim
        self.heads = heads
        self.ff = FeedForward(dim, mlp_dim)
        self.att = Attention(dim, heads = heads, dim_head = dim_head)
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head),
                FeedForward(dim, mlp_dim)
            ]))
   def exchange_info(self, x1=None, x2=None, x3=None, dim=512):
        if x3 == None:
            x_tmp_1 = x1[:,:,0:round(dim/3)]
            x_tmp_2 = x2[:, :, round(dim / 3):]
            x1_change = torch.cat([x_tmp_1, x_tmp_2], dim=2)
            x_tmp_1 = x2[:,:,0:round(dim/3)]
            x_tmp_2 = x1[:, :, round(dim / 3):]
            x2_change = torch.cat([x_tmp_1, x_tmp_2], dim=2)
            return x1_change, x2_change
        else :
            x_tmp_1 = x1[:,:, 0:round(dim/3)]
            x_tmp_2 = x2[:, :, round(dim / 3):2*round(dim / 3)]
            x_tmp_3 = x3[:, :, 2*round(dim / 3):]
            x1_change = torch.cat([x_tmp_1, x_tmp_2, x_tmp_3], dim=2)
            # x_tmp_1 = x1[:,:,0:round(dim/3)]
            # x_tmp_2 = x2[:, :, round(dim / 3):]
            # x2_change = torch.cat([x_tmp_1, x_tmp_2], dim=2)
            return x1_change
    def forward(self, x):
        for attn, ff in self.layers:
             x_dfc, x_alff, x_fa, x_sfc = x.chunk(4, dim = 1)
            x_dfc = self.exchange_info(x_dfc, x_alff, x_sfc, dim=self.dim)
            x_alff = self.exchange_info(x_alff, x_dfc, x_fa, dim=self.dim)
            x_fa = self.exchange_info(x_fa, x_dfc, x_alff, dim=self.dim)
            x_sfc = self.exchange_info(x_sfc, x_dfc, x_alff, dim=self.dim)
            x_dfc = attn(x_dfc) + x_dfc
            x_alff = attn(x_alff) + x_alff
            x_sfc = attn(x_sfc) + x_sfc
            x_fa = attn(x_fa) + x_fa
            x_dfc_sfc = x_dfc + x_sfc
            x_dfc_sfc_att = attn(x_dfc_alff)
            x_dfc_sfc = x_dfc_sfc + x_dfc_alff_att
            x_dfc_sfc_ff = ff(x_dfc_sfc) + x_dfc_sfc
            x_dfc_alff = x_dfc_sfc + x_dfc_sfc_ff
            x_fa_alff = x_alff + x_fa
            x_fa_alff_att = self.att(x_fa_alff)
            x_fa_alff = x_fa_alff + x_fa_alff_att
            x_fa_alff_ff = ff(x_fa_alff) + x_fa_alff
            x_fa_alff = x_fa_alff + x_fa_alff_ff
            x_dfc_sfc = attn([x_dfc_sfc,x_fa_alff_ff], cross =True) + x_dfc_sfc
            x_fa_alff = attn([x_fa_alff,x_dfc_sfc], cross =True) + x_fa_alff
            x_multi = x_fa_alff + x_dfc_sfc
            x_multi= attn(x_multi) + x_multi
            x_multi = ff(x_multi) + x_multi
            x = x_multi
        return x

class SimpleViT(nn.Module):
    def __init__(self, seq_len, patch_size, num_classes, dim, depth, heads, mlp_dim, channels=1, dim_head=16):
        super().__init__()
        self.conv_fusion = nn.Conv1d(in_channels=16, out_channels=1, kernel_size=1)
        self.bn_fusion = nn.BatchNorm1d(1)
        assert seq_len % patch_size == 0

        num_patches = seq_len // patch_size
        patch_dim = channels * patch_size

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (n p) -> b n (p c)', p = patch_size),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        self.transformer = Transformer(dim, depth, heads, mlp_dim, dim_head)

        self.to_latent = nn.Identity()
        self.linear_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, series):
        *_, n, dtype = *series.shape, series.dtype
        x = self.to_patch_embedding(series[:, 0:1, :])
        pe = posemb_sincos_1d(x)
        x = rearrange(x, 'b ... d -> b (...) d') + pe
        x_multi = x
        for modal in range(1,series.shape[1]):
            x_modal = self.to_patch_embedding(series[:,modal:modal+1,:])
            pe = posemb_sincos_1d(x_modal)
            x_modal = rearrange(x_modal, 'b ... d -> b (...) d') + pe
            x_multi = torch.cat([x_multi, x_modal], dim=1)

        x = self.transformer(x_multi)
        x = x.mean(dim = 1)
        x = self.to_latent(x)
        return self.linear_head(x)
class my_model_name(nn.Module):
    def  __init__(self, block, layers, opt, shortcut_type='B', num_classes=400, last_fc=True):
        super(my_model_name, self).__init__()
        self.avgpool_choose = avgpool_choose(opt)
        self.Resnet = ResNet(block, layers, opt, shortcut_type=shortcut_type, num_classes=num_classes)
        self.contrastive_loss = contrastive_loss(opt, opt.n_views)
        self.dfc_pyramid = dfc_pyramid(block, layers, opt, shortcut_type=shortcut_type, num_classes=num_classes)
        self.Transformer = SimpleViT(opt.seq_len, opt.patch_size, num_classes, opt.dim, opt.depth, opt.heads, opt.mlp_dim)
        self.weight_ce = opt.weight_ce
        self.weight_cl_fl = opt.weight_cl_fl
        self.weight_cl_fc = opt.weight_cl_fc
        self.opt = opt
        self.last_fc = last_fc
        if 'CN_MCI'==opt.category:
            weight_crossEntropy = torch.tensor(opt.cross_entropy_weights_CN_MCI)
        elif 'CN_MCI_SMC' == opt.category:
            weight_crossEntropy = torch.tensor(opt.cross_entropy_weights_CN_MCI_SMC)
        else:
            weight_crossEntropy = None
        self.criterion = torch.nn.CrossEntropyLoss(weight=weight_crossEntropy).cuda()
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        self.conv_fc = nn.Conv3d(512, 512, kernel_size=3, stride=(2, 2, 2),
                               padding=(1, 1, 1), bias=False)
        self.conv1D_fa_alff_1 = nn.Conv1d(in_channels=512, out_channels=64, kernel_size=3)
        self.bn_fa_alff_1 = nn.BatchNorm1d(64)
        self.conv1D_fa_alff_2 = nn.Conv1d(in_channels=64, out_channels=4, kernel_size=3)
        self.bn_fa_alff_2 = nn.BatchNorm1d(4)
        self.conv1D_fa_alff_3 = nn.Conv1d(in_channels=387, out_channels=256, kernel_size=3)
        self.bn_fa_alff_3 = nn.BatchNorm1d(256)
        self.conv1D_fa_alff_fusion= nn.Conv1d(in_channels=4, out_channels=1, kernel_size=1)
        self.bn_fa_alff_fusion = nn.BatchNorm1d(1)
        # self.conv1D_fa_alff_2 = nn.Conv1d(in_channels=4, out_channels=4, kernel_size=2)
        # self.conv1D_fa_alff_3 = nn.Conv1d(in_channels=4, out_channels=4, kernel_size=3)
        self.avgpool_fc = nn.AvgPool3d((6, 3, 1), stride=1)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        x_res = x.copy()
        fea_arr_fc = []
        fea_arr_local_dti = []
        x_array_list = list()
        for x in x_res[0]:
            # torch.cuda.empty_cache()
            shape_res_H = x.shape[2]
            # shape_res_W = x.shape[3]
            shape_res_T = x.shape[4]
            x = torch.where(torch.isinf(x), torch.full_like(x, 0), x)
            if shape_res_T == self.opt.sample_duration_dfc and shape_res_H == self.opt.sample_size1_fc:
                x = self.dfc_pyramid(x)
                # torch.var_mean(x, dim=4, keepdim=True)[1]
                # x = self.Resnet(x)
            else:
                # avgpool = self.avgpool_choose(x)
                x = self.Resnet(x)
            if shape_res_H == self.opt.sample_size1_fc:
                fea_arr_fc.append(x)
            if shape_res_H == self.opt.sample_size1_fmri:
                fea_arr_local_dti.append(x)
            # x = x.view(x.size(0), -1)
            x_array_list.append(x)
        x_array_list[1] = x_array_list[1].view(x_array_list[1].size(0), -1)
        x_array_list[0] = x_array_list[0].view(x_array_list[0].size(0), -1)
        x_array_list[2] = x_array_list[2].view(x_array_list[2].size(0), -1)
        x_array_list[3] = x_array_list[3].view(x_array_list[3].size(0), -1)
        # x_array_list[2] = fc_vector.view(fc_vector .size(0), -1)
        x_multi_add = torch.zeros(x_array_list[0].shape).cuda()
        x_multi_multiply = torch.ones(x_array_list[0].shape).cuda()
        for tensor in x_array_list:
            x_multi_add = torch.add(x_multi_add, tensor)
            x_multi_multiply = torch.multiply(x_multi_multiply, tensor)
            x_mid_multiply = torch.multiply(x_multi_add, x_multi_multiply)
            x_multi = torch.add(x_multi_add, x_mid_multiply)
        x_avg = x_multi_add/len(x_array_list)
        rand_arr_1 = random.sample(range(4),4)
        # rand_arr_1 = torch.tensor([0,1,2,3])
        rand_arr_2 = random.sample(range(4),4)
        # rand_arr_2 = torch.tensor([3,2,1,0])
        x_trans_1 = torch.cat([torch.unsqueeze(x_array_list[rand_arr_1[0]], dim=2),
                             torch.unsqueeze(x_array_list[rand_arr_1[1]], dim=2),
                             torch.unsqueeze(x_array_list[rand_arr_1[2]], dim=2),
                             torch.unsqueeze(x_array_list[rand_arr_1[3]], dim=2)
                             ], dim=2)
        x_trans_1 = torch.transpose(x_trans_1, 1, 2)
        x_t_1 = self.Transformer(x_trans_1)
        if len(fea_arr_fc) > 1 or len(fea_arr_local_dti) > 1:
            fea_arr_local_dti_fusion = [torch.multiply(x_array_list[0], x_array_list[2])
                , torch.add(x_array_list[0], x_array_list[2])]
            '''
            contrastive learning for FSA  and DSA
            '''
            loss_cl_fsa = self.contrastive_loss(fea_arr_local_dti_fusion)
            loss_cl_dsa = self.contrastive_loss([x_array_list[1], x_array_list[3]])
            loss_ce = self.criterion(x, x_t_1)
            loss = loss_ce + loss_cl_fc + loss_cl_local_dti
            return loss, x
        else:
            loss = nn.CrossEntropyLoss(x, x_res[0])
            return loss, x

def get_fine_tuning_parameters(model, ft_begin_index):
    if ft_begin_index == 0:
        return model.parameters()

    ft_module_names = []
    for i in range(ft_begin_index, 5):
        ft_module_names.append('layer{}'.format(ft_begin_index))
    ft_module_names.append('fc')

    parameters = []
    for k, v in model.named_parameters():
        for ft_module in ft_module_names:
            if ft_module in k:
                parameters.append({'params': v})
                break
        else:
            parameters.append({'params': v, 'lr': 0.0})

    return parameters

def my_model(**kwargs):
    model = my_model_name(BasicBlock, [1, 1, 1, 1], **kwargs)
    return model
def resnet10(**kwargs):
    """Constructs a ResNet-10 model.
    """
    model = ResNet(BasicBlock, [1, 1, 1, 1], **kwargs)
    return model

def resnet18(**kwargs):
    """Constructs a ResNet-18 model.
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    return model
