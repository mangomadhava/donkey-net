from torch import nn
import torch.nn.functional as F
import torch
from modules.movement_embedding import MovementEmbeddingModule
import math 

class DownBlock3D(nn.Module):
    """
    Simple block for processing video (encoder).
    """

    def __init__(self, in_features, out_features, norm=False, kernel_size=4):
        super(DownBlock3D, self).__init__()
        ka = kernel_size // 2
        kb = ka - 1 if kernel_size % 2 == 0 else ka

        self.conv = nn.Conv3d(in_channels=in_features, out_channels=out_features,
                              kernel_size=(1, kernel_size, kernel_size))
        if norm:
            self.norm = nn.InstanceNorm3d(out_features, affine=True)
        else:
            self.norm = None

    def forward(self, x):
        out = x
        out = self.conv(out)
        if self.norm:
            out = self.norm(out)
        out = F.leaky_relu(out, 0.2)
        out = F.avg_pool3d(out, (1, 2, 2))
        return out


class Discriminator(nn.Module):
    """
    Discriminator similar to Pix2Pix
    """

    def __init__(self, num_channels=3, num_kp=10, kp_variance=0.01, scale_factor=1,
                 block_expansion=64, num_blocks=4, max_features=512, kp_embedding_params=None, 
                 use_attention=False):

        super(Discriminator, self).__init__()

        if kp_embedding_params is not None:
            self.kp_embedding = MovementEmbeddingModule(num_kp=num_kp, kp_variance=kp_variance,
                                                        num_channels=num_channels,
                                                        **kp_embedding_params)
            embedding_channels = self.kp_embedding.out_channels
        else:
            self.kp_embedding = None
            embedding_channels = 0

        down_blocks = []
        for i in range(num_blocks):
            down_blocks.append(DownBlock3D(
                num_channels + embedding_channels if i == 0 else min(max_features, block_expansion * (2 ** i)),
                min(max_features, block_expansion * (2 ** (i + 1))),
                norm=(i != 0),
                kernel_size=4))

        self.use_attention = use_attention 

        self.down_blocks = nn.ModuleList(down_blocks)
        self.conv = nn.Conv3d(self.down_blocks[-1].conv.out_channels, out_channels=1, kernel_size=1)
        self.scale_factor = scale_factor
    
        if self.use_attention:
            reduction_ratio = 16 

            out_channels = [64, 128, 256, 256, 512] 
            k_list = [2 * math.floor(math.log2(C) * 0.25 + 0.25) + 1 for C in out_channels]  

            self.avg_pool = nn.AdaptiveAvgPool3d(1)
            self.eca_convs = nn.ModuleList([nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) //2, bias=False) for 
                k_size in k_list])
            
            self.spatial_gate_conv_1 = nn.ModuleList([nn.Conv3d(out_channel, out_channel // reduction_ratio, 
                                                    kernel_size=1) for out_channel in out_channels]) 
            self.spatial_gate_dil_conv_1 =  nn.ModuleList([nn.Conv3d(out_channel // reduction_ratio, 
                out_channel // reduction_ratio, kernel_size=3, dilation=2, padding=2) for out_channel in out_channels]) 

            self.spatial_gate_dil_conv_2 =  nn.ModuleList([nn.Conv3d(out_channel // reduction_ratio, 
                out_channel // reduction_ratio, kernel_size=3, dilation=4, padding=4) for out_channel in out_channels]) 

            self.spatial_gate_conv_2 = nn.ModuleList([nn.Conv3d(out_channel // reduction_ratio,1, kernel_size=1)
                                                    for out_channel in out_channels]) 
            self.batch_norm_1 = nn.ModuleList([nn.BatchNorm3d(out_channel // reduction_ratio) 
                                                    for out_channel in out_channels])
            self.batch_norm_2 = nn.ModuleList([nn.BatchNorm3d(out_channel // reduction_ratio) 
                                                    for out_channel in out_channels])
                   
    def generate_attention(self, x, i): 

        
        # print('x: ', x.shape) 

        # Create the channel attention map 
        avg_pool = self.avg_pool(x)

        # print('avg_pool: ', avg_pool.shape) 

        out1 = self.eca_convs[i](avg_pool.squeeze(-1).squeeze(-1).transpose(-1,-2)).transpose(-1,-2) \
                .unsqueeze(-1).unsqueeze(-1)
        
        # print('out1: ', out1.shape) 

        out2 = self.spatial_gate_conv_1[i](x)
        
        # print('out2: ', out2.shape) 


        out2 = self.spatial_gate_dil_conv_1[i](out2)

        # print('out2: ', out2.shape) 

        out2 = self.batch_norm_1[i](out2)

        # print('out2: ', out2.shape) 

        out2 = F.relu(out2)
        out2 = self.spatial_gate_dil_conv_2[i](out2)
        
        # print('out2: ', out2.shape) 

        out2 = self.batch_norm_2[i](out2)
        out2 = F.relu(out2)
        out2 = self.spatial_gate_conv_2[i](out2)

        # print('out2: ', out2.shape) 

        attn_sum = out1 * out2
        
        # print('attn_sum: ', attn_sum.shape) 

        attn_weights = F.sigmoid(attn_sum) + 1

        # print('attn_weights: ', attn_weights.shape) 


        return attn_weights


    def forward(self, x, kp_driving, kp_source):
        out_maps = [x]
    
        if self.scale_factor != 1:
            x = F.interpolate(x, scale_factor=(1, self.scale_factor, self.scale_factor))
 
        if self.kp_embedding:
            heatmap = self.kp_embedding(x, kp_driving, kp_source)
            out = torch.cat([x, heatmap], dim=1) 
        else:
            out = x
        for i, down_block in enumerate(self.down_blocks):
            if self.use_attention: 
                next_feats = (down_block(out))
                attended_features = self.generate_attention(next_feats, i)
                out_maps.append(attended_features * next_feats + next_feats)
                out = out_maps[-1]
                # print('out: ', out.shape)
            else:
                out_maps.append(down_block(out))
                out = out_maps[-1]
        out = self.conv(out)
        out_maps.append(out)
        return out_maps
