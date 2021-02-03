import torch
from torch import nn
import torch.nn.functional as F

from modules.util import Encoder, Decoder, ResBlock3D
from modules.dense_motion_module import DenseMotionModule, IdentityDeformation
from modules.movement_embedding import MovementEmbeddingModule


class MotionTransferGenerator(nn.Module):
    """
    Motion transfer generator. That Given a keypoints and an appearance trying to reconstruct the target frame.
    Produce 2 versions of target frame, one warped with predicted optical flow and other refined.
    """

    def __init__(self, num_channels, num_kp, kp_variance, block_expansion, max_features, num_blocks, num_refinement_blocks,
                 dense_motion_params=None, kp_embedding_params=None, interpolation_mode='nearest', 
                 use_generator_attention=False, use_dmm_attention=False):

        super(MotionTransferGenerator, self).__init__()

        self.appearance_encoder = Encoder(block_expansion, in_features=num_channels, max_features=max_features,
                                          num_blocks=num_blocks)

        if kp_embedding_params is not None:
            self.kp_embedding_module = MovementEmbeddingModule(num_kp=num_kp, kp_variance=kp_variance,
                                                               num_channels=num_channels, **kp_embedding_params)
            embedding_features = self.kp_embedding_module.out_channels
        else:
            self.kp_embedding_module = None
            embedding_features = 0

        if dense_motion_params is not None:
            self.dense_motion_module = DenseMotionModule(num_kp=num_kp, kp_variance=kp_variance,
                                                         num_channels=num_channels,
                                                         **dense_motion_params, use_attention=use_dmm_attention)
        else:
            self.dense_motion_module = IdentityDeformation()

        self.use_attention = 'generator' if use_generator_attention else None 
        self.video_decoder = Decoder(block_expansion=block_expansion, in_features=num_channels,
                                     out_features=num_channels, max_features=max_features, num_blocks=num_blocks,
                                     additional_features_for_block=embedding_features,
                                     use_last_conv=False, use_attention=self.use_attention)

        self.refinement_module = torch.nn.Sequential()
        in_features = block_expansion + num_channels + embedding_features
        for i in range(num_refinement_blocks):
            self.refinement_module.add_module('r' + str(i),
                                              ResBlock3D(in_features, kernel_size=(1, 3, 3), padding=(0, 1, 1)))
        self.refinement_module.add_module('conv-last', nn.Conv3d(in_features, num_channels, kernel_size=1, padding=0))
        self.interpolation_mode = interpolation_mode

    def deform_input(self, inp, deformations_absolute):
        bs, d, h_old, w_old, _ = deformations_absolute.shape
        _, _, _, h, w = inp.shape
        deformations_absolute = deformations_absolute.permute(0, 4, 1, 2, 3)
        deformation = F.interpolate(deformations_absolute, size=(d, h, w), mode=self.interpolation_mode)
        deformation = deformation.permute(0, 2, 3, 4, 1)
        deformed_inp = F.grid_sample(inp, deformation)
        return deformed_inp

    def forward(self, source_image, kp_driving, kp_source):
        
        # print(source_image.shape) 
        appearance_skips = self.appearance_encoder(source_image)

        
        # print(appearance_skips[0].shape, len(appearance_skips)) 
        deformations_absolute = self.dense_motion_module(source_image=source_image, kp_driving=kp_driving,
                                                         kp_source=kp_source)

        # print(deformations_absolute.shape) 
        deformed_skips = [self.deform_input(skip, deformations_absolute) for skip in appearance_skips]

        # print(deformed_skips[0].shape, len(deformed_skips)) 

        if self.kp_embedding_module is not None:
            d = kp_driving['mean'].shape[1]
            # print(d)
            movement_embedding = self.kp_embedding_module(source_image=source_image, kp_driving=kp_driving,
                                                          kp_source=kp_source)
            # print(movement_embedding.shape)
            kp_skips = [F.interpolate(movement_embedding, size=(d,) + skip.shape[3:], mode=self.interpolation_mode) for skip in appearance_skips]
            # print(kp_skips[0].shape, len(kp_skips)) 
            skips = [torch.cat([a, b], dim=1) for a, b in zip(deformed_skips, kp_skips)]
        else:
            skips = deformed_skips

        # print(skips[0].shape, len(skips))
        video_deformed = self.deform_input(source_image, deformations_absolute)
        
        # print(video_deformed.shape)
        
        video_prediction = self.video_decoder(skips)
        # print(video_prediction.shape)
        video_prediction = self.refinement_module(video_prediction)
        # print(video_prediction.shape)
        video_prediction = torch.sigmoid(video_prediction)
        # print(video_prediction.shape)

        return {"video_prediction": video_prediction, "video_deformed": video_deformed}
