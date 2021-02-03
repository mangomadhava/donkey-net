import matplotlib 

matplotlib.use('Agg')
import matplotlib.pyplot as plt 

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
import os 

from tqdm import tqdm 

from sync_batchnorm import DataParallelWithCallback
from frames_dataset import read_video
from augmentation import VideoToTensor

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", required=True, help="path to config")
    parser.add_argument("--out_file", default='demo.gif', help="Path to out file")
    parser.add_argument("--driving_directory", required=True, help='Path to driving images')
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint")
    parser.add_argument("--cpu", dest="cpu", action="store_true", help="Use cpu")
    parser.add_argument("--image_shape", default=(128, 128), type=lambda x: tuple([int(a) for a in x.split(',')]),
                        help="Image shape")


    parser.add_argument("--name" , required=True, help="dataset name", type=str) 
    parser.add_argument("--visualize", default=False, action="store_true", help="Visualize the pose on the given frame") 

    parser.set_defaults(cpu=False)

    opt = parser.parse_args()

    with open(opt.config) as f:
        config = yaml.load(f)
        blocks_discriminator = config['model_params']['discriminator_params']['num_blocks']
        assert len(config['train_params']['loss_weights']['reconstruction']) == blocks_discriminator + 1

    generator = MotionTransferGenerator(**config['model_params']['generator_params'],
                                        **config['model_params']['common_params'])
    if not opt.cpu:
        generator.cuda()

    kp_detector = KPDetector(**config['model_params']['kp_detector_params'],
                             **config['model_params']['common_params'])
    if not opt.cpu:
        kp_detector = kp_detector.cuda()

    Logger.load_cpk(opt.checkpoint, generator=generator, kp_detector=kp_detector, use_cpu=True)

    vis = Visualizer()

    if not opt.cpu: 
        generator = DataParallelWithCallback(generator)
        kp_detector = DataParallelWithCallback(kp_detector)

    generator.eval()
    kp_detector.eval()

    '''
    Logic: The goal of this module is to essentially loop through all of the GIFs in a directory and then 
    extract the pose points for the first frame of the GIF for each GIF. This allows for an alignment based
    on only the first frame. 
    
    TODO: Extend this to extract poses from the driving video to then
    obtain poses at each frame for alignment. 

    '''
    with torch.no_grad():
        
        # This dictionary stores the initial pose for each of the GIFs
        poses_dict = {}

        for img_name in tqdm(os.listdir(opt.driving_directory)):
            
            path_name = opt.driving_directory + img_name
            driving_video = VideoToTensor()(read_video(path_name, opt.image_shape + (3,)))['video']
           
            driving_video = torch.from_numpy(driving_video).unsqueeze(0)

            cat_dict = lambda l, dim: {k: torch.cat([v[k] for v in l], dim=dim) for k in l[0]}
            d = driving_video.shape[2] 
            
            kp_driving = cat_dict([kp_detector(driving_video[:,:,i:(i+1)]) for i in range(d)], dim=1) 

            poses_dict[img_name] = kp_driving 
        
        # Dump the poses dict 
        pickle.dump(poses_dict, open('./driving_video_poses/{}_poses.pkl'.format(opt.name), 'wb')) 



