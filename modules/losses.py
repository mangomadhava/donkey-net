import torch
import torch.nn as nn 
from torch.autograd import grad 
import torch.nn.functional as F 
from torchvision import models 
import numpy as np 

class VGG19(nn.Module): 
    '''
    VGG19 Network for Perceptual Loss 
    ''' 
    def __init__(self, requires_grad=False): 
        super(VGG19, self).__init__()
        vgg_pretrained_features = models.vgg19(pretrained=True).features 
        blocks = []
        blocks.append(vgg_pretrained_features[:2].eval())
        blocks.append(vgg_pretrained_features[2:7].eval())
        blocks.append(vgg_pretrained_features[7:12].eval())
        blocks.append(vgg_pretrained_features[12:21].eval())
        blocks.append(vgg_pretrained_features[21:30].eval())

        self.blocks = torch.nn.ModuleList(blocks) 
        self.mean = torch.nn.Parameter(torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1))
        self.std = torch.nn.Parameter(torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1))
        
        if not requires_grad: 
            for param in self.parameters():
                param.requires_grad = False 

    def forward(self, x, y, weights): 
        x = (x - self.mean) / self.std 
        y = (y - self.mean) / self.std 
        loss = 0.0 
        for i, block in enumerate(self.blocks): 
            x = block(x)
            y = block(y)
            loss += F.l1_loss(x, y) * weights[i] 
        return loss 

def mean_batch(val):
    return val.view(val.shape[0], -1).mean(-1)

def reconstruction_loss(prediction, target, weight):
    if weight == 0:
        return 0
    return weight * mean_batch(torch.abs(prediction - target))


def generator_gan_loss(discriminator_maps_generated, weight):
    scores_generated = discriminator_maps_generated[-1]
    score = (1 - scores_generated) ** 2
    return weight * mean_batch(score)


def discriminator_gan_loss(discriminator_maps_generated, discriminator_maps_real, weight):
    scores_real = discriminator_maps_real[-1]
    scores_generated = discriminator_maps_generated[-1]
    score = (1 - scores_real) ** 2 + scores_generated ** 2
    return weight * mean_batch(score)


def generator_loss_names(loss_weights):
    loss_names = []
    if loss_weights['reconstruction_deformed'] != 0:
        loss_names.append("rec_def")

    if loss_weights['reconstruction'] is not None:
        for i, _ in enumerate(loss_weights['reconstruction']):
            if loss_weights['reconstruction'][i] == 0:
                continue
            loss_names.append("layer-%s_rec" % i)

    loss_names.append("gen_gan")
    return loss_names


def discriminator_loss_names():
    return ['disc_gan']


def generator_loss(discriminator_maps_generated, discriminator_maps_real, video_deformed, loss_weights):
    loss_values = []
    if loss_weights['reconstruction_deformed'] != 0:
        loss_values.append(reconstruction_loss(discriminator_maps_real[0], video_deformed,
                                               loss_weights['reconstruction_deformed']))

    if loss_weights['reconstruction'] != 0:
        for i, (a, b) in enumerate(zip(discriminator_maps_real[:-1], discriminator_maps_generated[:-1])):
            if loss_weights['reconstruction'][i] == 0:
                continue
            loss_values.append(reconstruction_loss(b, a, weight=loss_weights['reconstruction'][i]))

    loss_values.append(generator_gan_loss(discriminator_maps_generated, weight=loss_weights['generator_gan']))

    return loss_values


def discriminator_loss(discriminator_maps_generated, discriminator_maps_real, loss_weights):
    loss_values = [discriminator_gan_loss(discriminator_maps_generated, discriminator_maps_real,
                                          loss_weights['discriminator_gan'])]

    return loss_values

