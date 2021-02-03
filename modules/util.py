from torch import nn

import torch.nn.functional as F
import torch

import numpy as np
from sync_batchnorm import SynchronizedBatchNorm3d as BatchNorm3d

import math 

def compute_image_gradient(image, padding=0):
    bs, c, h, w = image.shape

    sobel_x = torch.from_numpy(np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])).type(image.type())
    filter = sobel_x.unsqueeze(0).repeat(c, 1, 1, 1)
    grad_x = F.conv2d(image, filter, groups=c, padding=padding)
    grad_x = grad_x

    sobel_y = torch.from_numpy(np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])).type(image.type())
    filter = sobel_y.unsqueeze(0).repeat(c, 1, 1, 1)
    grad_y = F.conv2d(image, filter, groups=c, padding=padding)
    grad_y = grad_y

    return torch.cat([grad_x, grad_y], dim=1)


def make_coordinate_grid(spatial_size, type):
    """
    Create a meshgrid [-1,1] x [-1,1] of given spatial_size.
    """
    h, w = spatial_size
    x = torch.arange(w).type(type)
    y = torch.arange(h).type(type)

    x = (2 * (x / (w - 1)) - 1)
    y = (2 * (y / (h - 1)) - 1)

    yy = y.view(-1, 1).repeat(1, w)
    xx = x.view(1, -1).repeat(h, 1)

    meshed = torch.cat([xx.unsqueeze_(2), yy.unsqueeze_(2)], 2)

    return meshed


class ResBlock3D(nn.Module):
    """
    Res block, preserve spatial resolution.
    """

    def __init__(self, in_features, kernel_size, padding):
        super(ResBlock3D, self).__init__()
        self.conv1 = nn.Conv3d(in_channels=in_features, out_channels=in_features, kernel_size=kernel_size,
                               padding=padding)
        self.conv2 = nn.Conv3d(in_channels=in_features, out_channels=in_features, kernel_size=kernel_size,
                               padding=padding)
        self.norm1 = BatchNorm3d(in_features, affine=True)
        self.norm2 = BatchNorm3d(in_features, affine=True)

    def forward(self, x):
        out = x
        out = self.norm1(x)
        out = F.relu(out)
        out = self.conv1(out)
        out = self.norm2(out)
        out = F.relu(out)
        out = self.conv2(out)
        out += x
        return out


class UpBlock3D(nn.Module):
    """
    Simple block for processing video (decoder).
    """

    def __init__(self, in_features, out_features, kernel_size=3, padding=1):
        super(UpBlock3D, self).__init__()

        self.conv = nn.Conv3d(in_channels=in_features, out_channels=out_features, kernel_size=kernel_size,
                              padding=padding)
        self.norm = BatchNorm3d(out_features, affine=True)

    def forward(self, x):
        out = F.interpolate(x, scale_factor=(1, 2, 2))
        out = self.conv(out)
        out = self.norm(out)
        out = F.relu(out)
        return out


class DownBlock3D(nn.Module):
    """
    Simple block for processing video (encoder).
    """

    def __init__(self, in_features, out_features, kernel_size=3, padding=1):
        super(DownBlock3D, self).__init__()
        self.conv = nn.Conv3d(in_channels=in_features, out_channels=out_features, kernel_size=kernel_size,
                              padding=padding)
        self.norm = BatchNorm3d(out_features, affine=True)
        self.pool = nn.AvgPool3d(kernel_size=(1, 2, 2))

    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        out = F.relu(out)
        out = self.pool(out)
        return out


class SameBlock3D(nn.Module):
    """
    Simple block with group convolution.
    """

    def __init__(self, in_features, out_features, groups=None, kernel_size=3, padding=1):
        super(SameBlock3D, self).__init__()
        self.conv = nn.Conv3d(in_channels=in_features, out_channels=out_features,
                              kernel_size=kernel_size, padding=padding, groups=groups)
        self.norm = BatchNorm3d(out_features, affine=True)

    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        out = F.relu(out)
        return out


class Encoder(nn.Module):
    """
    Hourglass Encoder
    """

    def __init__(self, block_expansion, in_features, num_blocks=3, max_features=256, temporal=False):
        super(Encoder, self).__init__()

        down_blocks = []

        kernel_size = (3, 3, 3) if temporal else (1, 3, 3)
        padding = (1, 1, 1) if temporal else (0, 1, 1)
        for i in range(num_blocks):
            down_blocks.append(DownBlock3D(in_features if i == 0 else min(max_features, block_expansion * (2 ** i)),
                                           min(max_features, block_expansion * (2 ** (i + 1))),
                                           kernel_size=kernel_size, padding=padding))
        self.down_blocks = nn.ModuleList(down_blocks)

    def forward(self, x):
        outs = [x]

        for down_block in self.down_blocks:
            outs.append(down_block(outs[-1]))
        return outs


class Decoder(nn.Module):
    """
    Hourglass Decoder
    """

    def __init__(self, block_expansion, in_features, out_features, num_blocks=3, max_features=256, temporal=False,
                 additional_features_for_block=0, use_last_conv=True, use_attention=None):
        super(Decoder, self).__init__()
        kernel_size = (3, 3, 3) if temporal else (1, 3, 3)
        padding = (1, 1, 1) if temporal else (0, 1, 1)

        up_blocks = []

        for i in range(num_blocks)[::-1]:
            up_blocks.append(UpBlock3D((1 if i == num_blocks - 1 else 2) * min(max_features, block_expansion * (
                2 ** (i + 1))) + additional_features_for_block,
                                       min(max_features, block_expansion * (2 ** i)),
                                       kernel_size=kernel_size, padding=padding))

        self.up_blocks = nn.ModuleList(up_blocks)
        if use_last_conv:
            self.conv = nn.Conv3d(in_channels=block_expansion + in_features + additional_features_for_block,
                                  out_channels=out_features, kernel_size=kernel_size, padding=padding)
        else:
            self.conv = None

        self.use_attention = use_attention

        if self.use_attention == 'dmm' or self.use_attention == 'generator': 

            # Channel attention module 
            self.avg_pool = nn.AdaptiveAvgPool3d((1,1,1))
        
            if self.use_attention == 'dmm':
                input_channels = [512, 256, 128, 64, 32] 
            elif self.use_attention == 'generator': 
                input_channels = [1024, 512, 256, 128, 64, 32]  

            k_list = [2 * math.floor(math.log2(C) * 0.25 + 0.25) + 1 for C in input_channels]

            self.eca_convs = nn.ModuleList([nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1)//2, bias=False) for 
                                            k_size in k_list]) 
            # Spatial attention module (apply a 3D convolution on the concatenated 
            self.spatial_conv = nn.Conv3d(2, 1, kernel_size=7, stride=1, padding=3)

            
    def generate_attention(self, x, i):

        ''' This is a channel spatial attention module that first uses ECA and then passes the attended 
            feature map to the spatial module where the avg/max spatial information is obtained over 
            all channels and then passed to a sigmoid to get the final attended fmaps. '''
       
        # print('x: ', x.shape) 
        avg_pool = self.avg_pool(x) 

        # print('avg_pool: ', avg_pool.shape)

        # This returns a tuple for each sample in the batch which is strange but it should be alright? 
        out = self.eca_convs[i](avg_pool.squeeze(-1).squeeze(-1).transpose(-1,-2)).transpose(-1,-2) \
                                                                                  .unsqueeze(-1).unsqueeze(-1)
        # print('out: ', out.shape) 
        
        scale = F.sigmoid(out) 
        channel_attended_fmap = scale * x 

        # print('scale, cfmap: ', scale.shape, channel_attended_fmap.shape)

        avg_pool_spatial = torch.mean(channel_attended_fmap, dim=1) 
        max_pool_spatial, _ = torch.max(channel_attended_fmap, dim=1)

        # print('spatial: ', avg_pool_spatial.shape, max_pool_spatial.shape)

        spatial_pool = torch.cat([avg_pool_spatial, max_pool_spatial], dim=1)
        scale = F.sigmoid(self.spatial_conv(spatial_pool.unsqueeze(2)))  

        # print('scale: ', spatial_pool.shape, scale.shape) 

        attended_fmap = scale * channel_attended_fmap

        # print('final: ', attended_fmap.shape)

        return attended_fmap 

    def forward(self, x):
        out = x.pop()
        # print(out.shape)

        for i, up_block in enumerate(self.up_blocks):

            out = up_block(out)
            # print('up: ', out.shape)

            if self.use_attention:
                attended_fmap = self.generate_attention(out, i) 
                out = torch.cat([attended_fmap + out, x.pop()], dim=1)
            else:
                out = torch.cat([out, x.pop()], dim=1)

        if self.conv is not None:
            return self.conv(out)
        else:
            return out


class Hourglass(nn.Module):
    """
    Hourglass architecture.
    """

    def __init__(self, block_expansion, in_features, out_features, num_blocks=3, max_features=256, temporal=False, use_attention=False):
        super(Hourglass, self).__init__()

        self.use_attention = 'dmm' if use_attention else None 
        self.encoder = Encoder(block_expansion, in_features, num_blocks, max_features, temporal=temporal)
        self.decoder = Decoder(block_expansion, in_features, out_features, num_blocks, max_features, temporal=temporal, 
                                use_attention=self.use_attention)

    def forward(self, x):
        return self.decoder(self.encoder(x))


def matrix_inverse(batch_of_matrix, eps=0):
    if eps != 0:
        init_shape = batch_of_matrix.shape
        a = batch_of_matrix[..., 0, 0].unsqueeze(-1)
        b = batch_of_matrix[..., 0, 1].unsqueeze(-1)
        c = batch_of_matrix[..., 1, 0].unsqueeze(-1)
        d = batch_of_matrix[..., 1, 1].unsqueeze(-1)

        det = a * d - b * c
        out = torch.cat([d, -b, -c, a], dim=-1)
        eps = torch.tensor(eps).type(out.type())
        out /= det.max(eps)

        return out.view(init_shape)
    else:
        b_mat = batch_of_matrix
        eye = b_mat.new_ones(b_mat.size(-1)).diag().expand_as(b_mat)
        b_inv, _ = torch.gesv(eye, b_mat)
        return b_inv


def matrix_det(batch_of_matrix):
    a = batch_of_matrix[..., 0, 0].unsqueeze(-1)
    b = batch_of_matrix[..., 0, 1].unsqueeze(-1)
    c = batch_of_matrix[..., 1, 0].unsqueeze(-1)
    d = batch_of_matrix[..., 1, 1].unsqueeze(-1)

    det = a * d - b * c
    return det


def matrix_trace(batch_of_matrix):
    a = batch_of_matrix[..., 0, 0].unsqueeze(-1)
    d = batch_of_matrix[..., 1, 1].unsqueeze(-1)

    return a + d


def smallest_singular(batch_of_matrix):
    a = batch_of_matrix[..., 0, 0].unsqueeze(-1)
    b = batch_of_matrix[..., 0, 1].unsqueeze(-1)
    c = batch_of_matrix[..., 1, 0].unsqueeze(-1)
    d = batch_of_matrix[..., 1, 1].unsqueeze(-1)

    s1 = a ** 2 + b ** 2 + c ** 2 + d ** 2
    s2 = (a ** 2 + b ** 2 - c ** 2 - d ** 2) ** 2
    s2 = torch.sqrt(s2 + 4 * (a * c + b * d) ** 2)

    norm = torch.sqrt((s1 - s2) / 2)
    return norm
