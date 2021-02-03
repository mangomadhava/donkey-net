import os
from skimage import io, img_as_float32
from skimage.color import gray2rgb
from sklearn.model_selection import train_test_split
from imageio import mimread

import numpy as np
from torch.utils.data import Dataset
import pandas as pd

from augmentation import AllAugmentationTransform, VideoToTensor


def get_images(filename):
    image_names = np.genfromtxt(filename, dtype=str)
    return image_names 

def read_video(name, image_shape):
    if name.lower().endswith('.png') or name.lower().endswith('.jpg'):
        image = io.imread(name)

        if len(image.shape) == 2 or image.shape[2] == 1:
            image = gray2rgb(image)

        if image.shape[2] == 4:
            image = image[..., :3]

        image = img_as_float32(image)

        video_array = np.moveaxis(image, 1, 0)

        video_array = video_array.reshape((-1,) + image_shape)
        video_array = np.moveaxis(video_array, 1, 2)
    elif name.lower().endswith('.gif') or name.lower().endswith('.mp4') or name.lower().endswith('.mov'):
        video = np.array(mimread(name))
        if len(video.shape) == 3:
            video = np.array([gray2rgb(frame) for frame in video])
        if video.shape[-1] == 4:
            video = video[..., :3]
        video_array = img_as_float32(video)
    else:
        raise Exception("Unknown file extensions  %s" % name)

    return video_array


class FramesDataset(Dataset):
    """Dataset of videos, videos can be represented as an image of concatenated frames, or in '.mp4','.gif' format"""

    def __init__(self, root_dir, augmentation_params, image_shape=(64, 64, 3), is_train=True,
                 random_seed=0, pairs_list=None, transform=None):
        self.root_dir = root_dir
        self.images = os.listdir(root_dir)
        self.image_shape = tuple(image_shape)
        self.pairs_list = pairs_list

        if os.path.exists(os.path.join(root_dir, 'train')):
            assert os.path.exists(os.path.join(root_dir, 'test'))
            print("Use predefined train-test split.")
            train_images = os.listdir(os.path.join(root_dir, 'train'))
            test_images = os.listdir(os.path.join(root_dir, 'test'))
            self.root_dir = os.path.join(self.root_dir, 'train' if is_train else 'test')
        else:
            print("Use random train-test split.")
            train_images, test_images = train_test_split(self.images, random_state=random_seed, test_size=0.2)

        if is_train:
            self.images = train_images
        else:
            self.images = test_images

        if transform is None:
            if is_train:
                self.transform = AllAugmentationTransform(**augmentation_params)
            else:
                self.transform = VideoToTensor()
        else:
            self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.images[idx])

        video_array = read_video(img_name, image_shape=self.image_shape)

        print('Video Array Shape: ', video_array.shape) 

        out = self.transform(video_array)
        # add names
        out['name'] = os.path.basename(img_name)

        return out

''' 
This is the custom dataset for handling pairs of images for different GIFs (Moving-GIF Dataset)
'''

class PairedFramesDataset(Dataset):
    """Dataset of videos, videos can be represented as an image of concatenated frames, or in '.mp4','.gif' format"""

    def __init__(self, root_dir, augmentation_params, is_train=True, image_shape=(64, 64, 3),
                 random_seed=0, transform=None):
        self.root_dir = root_dir
        self.images = os.listdir(root_dir)
        self.image_shape = tuple(image_shape) 

        if os.path.exists(os.path.join(root_dir, 'train')):
            assert os.path.exists(os.path.join(root_dir, 'test'))
            assert os.path.isfile(os.path.join(root_dir, 'train_pairing.txt')) 
            assert os.path.isfile(os.path.join(root_dir, 'test_pairing.txt')) 

            print("Use predefined train-test split.")
            train_images = os.listdir(os.path.join(root_dir, 'train'))
            test_images = os.listdir(os.path.join(root_dir, 'test'))
            self.root_dir = os.path.join(self.root_dir, 'train' if is_train else 'test')
        else:
            print("Use random train-test split.")
            train_images, test_images = train_test_split(self.images, random_state=random_seed, test_size=0.2)

        if is_train:
            self.images = train_images
            self.pairs_list = np.genfromtxt(os.path.join(root_dir, 'train_pairing.txt'), dtype=str)
        else:
            self.images = test_images
            self.pairs_list = np.genfromtxt(os.path.join(root_dir, 'test_pairing.txt'), dtype=str)

        if transform is None:
            self.transform = VideoToTensor()
        else:
            self.transform = transform

    def __len__(self):
        return len(self.pairs_list)

    def __getitem__(self, idx):
        
        gif1_name = os.path.join(self.root_dir, str(self.pairs_list[idx][2]))
        gif2_name = os.path.join(self.root_dir, str(self.pairs_list[idx][3]))
   
        gif1_video = read_video(gif1_name, image_shape=self.image_shape)
        gif2_video = read_video(gif2_name, image_shape=self.image_shape)

        # print('Vidoe Array: ', gif1_video.shape, gif2_video.shape)

        gif1_video = np.expand_dims(gif1_video, 2).transpose((0,4,2,1,3))
        gif2_video = np.expand_dims(gif2_video, 2).transpose((0,4,2,1,3))

        assert gif1_video.shape == gif2_video.shape 

        # add names
        out = {} 
        out['name'] = self.pairs_list[idx][2][:-4]

        # Generate a paired dataset (src_A, src_B, gt_A, gt_B) 
        frame_count = gif1_video.shape[0]
        num_frames_to_select = 3

        if num_frames_to_select > frame_count: 
            source_A_idx = 0
            source_B_idx = 0 
            driving_idx = 1 
        else:
            selected_index = np.sort(np.random.choice(range(frame_count), replace=False,
                size = num_frames_to_select))
            source_A_idx = selected_index[0]
            source_B_idx = selected_index[1]
            driving_idx = selected_index[2]

        out['src_A'] = gif1_video[source_A_idx, :, :, :, :] 
        out['src_B'] = gif2_video[source_B_idx, :, :, :, :]

        out['driving_A'] = gif1_video[driving_idx, :, :, :, :]
        out['driving_B'] = gif2_video[driving_idx, :, :, :, :]  
            
        return out

class PairedDataset(Dataset):
    """
    Dataset of pairs for transfer.
    """

    def __init__(self, initial_dataset, number_of_pairs, seed=0):
        self.initial_dataset = initial_dataset
        pairs_list = self.initial_dataset.pairs_list

        np.random.seed(seed)

        if pairs_list is None:
            max_idx = min(number_of_pairs, len(initial_dataset))
            nx, ny = max_idx, max_idx
            xy = np.mgrid[:nx, :ny].reshape(2, -1).T
            number_of_pairs = min(xy.shape[0], number_of_pairs)
            self.pairs = xy.take(np.random.choice(xy.shape[0], number_of_pairs, replace=False), axis=0)
        else:
            images = self.initial_dataset.images
            name_to_index = {name: index for index, name in enumerate(images)}
            pairs = pd.read_csv(pairs_list)
            pairs = pairs[np.logical_and(pairs['source'].isin(images), pairs['driving'].isin(images))]

            number_of_pairs = min(pairs.shape[0], number_of_pairs)
            self.pairs = []
            self.start_frames = []
            for ind in range(number_of_pairs):
                self.pairs.append(
                    (name_to_index[pairs['driving'].iloc[ind]], name_to_index[pairs['source'].iloc[ind]]))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        pair = self.pairs[idx]
        first = self.initial_dataset[pair[0]]
        second = self.initial_dataset[pair[1]]
        first = {'driving_' + key: value for key, value in first.items()}
        second = {'source_' + key: value for key, value in second.items()}

        return {**first, **second}
