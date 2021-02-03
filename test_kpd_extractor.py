import matplotlib 

matplotlib.use('Agg')

import torch
from transfer import transfer_one
import yaml
from logger import Logger, Visualizer 

from argparse import ArgumentParser

from modules.generator import MotionTransferGenerator
from modules.keypoint_detector import KPDetector

import imageio
import pickle 

import numpy as np

from sync_batchnorm import DataParallelWithCallback
from frames_dataset import read_video
from augmentation import VideoToTensor

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", required=True, help="path to config")
    parser.add_argument("--out_file", default='demo.gif', help="Path to out file")
    parser.add_argument("--driving_video", default='sup-mat/driving.png', help="Path to driving video")
    parser.add_argument("--source_image", default="sup-mat/source.png", help='Path to source image')
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint")
    parser.add_argument("--use_dmm_attention", action='store_true') 
    parser.add_argument("--use_generator_attention", action='store_true') 
    parser.add_argument("--cpu", dest="cpu", action="store_true", help="Use cpu")
    parser.add_argument("--image_shape", default=(128, 128), type=lambda x: tuple([int(a) for a in x.split(',')]),
                        help="Image shape")
    parser.set_defaults(cpu=False)

    opt = parser.parse_args()

    with open(opt.config) as f:
        config = yaml.load(f)
        blocks_discriminator = config['model_params']['discriminator_params']['num_blocks']
        assert len(config['train_params']['loss_weights']['reconstruction']) == blocks_discriminator + 1

    generator = MotionTransferGenerator(**config['model_params']['generator_params'],
                                        **config['model_params']['common_params'], 
                                        use_dmm_attention=opt.use_dmm_attention,
                                        use_generator_attention=opt.use_generator_attention)
 #   if not opt.cpu:
 #       generator.cuda()

    kp_detector = KPDetector(**config['model_params']['kp_detector_params'],
                             **config['model_params']['common_params'])
 #   if not opt.cpu:
 #       kp_detector = kp_detector.cuda()

    Logger.load_cpk(opt.checkpoint, generator=generator, kp_detector=kp_detector, use_cpu=False)

    vis = Visualizer()

    # generator = DataParallelWithCallback(generator)
    # kp_detector = DataParallelWithCallback(kp_detector)

    generator.eval()
    kp_detector.eval()

    with torch.no_grad():
        driving_video = VideoToTensor()(read_video(opt.driving_video, opt.image_shape + (3,)))['video']
        source_image = VideoToTensor()(read_video(opt.source_image, opt.image_shape + (3,)))['video'][:, :1]
        print(source_image.shape)
        
        driving_video = torch.from_numpy(driving_video).unsqueeze(0)
        source_image = torch.from_numpy(source_image).unsqueeze(0)

        out = transfer_one(generator, kp_detector, source_image, driving_video, config['transfer_params'])
        
        '''
        # Pickle the out 
        f = open('keypoints.pkl', 'wb')
        pickle.dump(out, f) 
        f.close()
        '''

        img = vis.visualize_transfer(driving_video, source_image, out) 
        print(type(img), img.shape)

        # Save image 
        imageio.mimsave('test_image2.gif', img)
        
        



