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
    parser.add_argument("--source_directory", required=True, help='Path to source image')
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

        for img_name in tqdm(os.listdir(opt.source_directory)):
            
            path_name = opt.source_directory + img_name
            source_image = VideoToTensor()(read_video(path_name, 
                opt.image_shape + (3,)))['video'][:, :1]
           
            print(source_image.shape) 
            source_image = torch.from_numpy(source_image).unsqueeze(0)

            #Extract the mean of the keypoints 
            mean = kp_detector(source_image)['mean'].data.cpu().numpy() 
            # Apply the transformation 
            key_points = 128 * (mean + 1) / 2 
            
            poses_dict[img_name[:-4]] = key_points 

            if opt.visualize:
                # img = vis.visualize_initial_pose(source_image, mean) 
                
                img = plt.imread(path_name)
                print(img.shape) 
                plt.imshow(img)
                
                # Save image 
                if not os.path.isdir('./vis/{}/'.format(opt.name)):
                    os.mkdir('./vis/{}'.format(opt.name))
                imageio.mimsave('./vis/{}/{}_with_kp.gif'.format(opt.name, img_name), img)
        
        # Dump the poses dict 
        pickle.dump(poses_dict, open('./single_image_poses/{}_initial_poses.pkl'.format(opt.name), 'wb')) 



