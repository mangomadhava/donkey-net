from tqdm import trange

import torch
from torch import nn 

from torch.utils.data import DataLoader

from logger import Logger
from modules.losses import generator_loss, discriminator_loss, generator_loss_names, discriminator_loss_names, VGG19

from torch.optim.lr_scheduler import MultiStepLR

from sync_batchnorm import DataParallelWithCallback

from tensorboardX import SummaryWriter

import os 

def split_kp(kp_joined, detach=False):
    if detach:
        kp_video = {k: v[:, 1:].detach() for k, v in kp_joined.items()}
        kp_appearance = {k: v[:, :1].detach() for k, v in kp_joined.items()}
    else:
        kp_video = {k: v[:, 1:] for k, v in kp_joined.items()}
        kp_appearance = {k: v[:, :1] for k, v in kp_joined.items()}
    return {'kp_driving': kp_video, 'kp_source': kp_appearance}


class GeneratorFullModel(torch.nn.Module):
    """
    Merge all generator related updates into single model for better multi-gpu usage
    """

    def __init__(self, kp_extractor, generator, discriminator, train_params):
        super(GeneratorFullModel, self).__init__()
        self.kp_extractor = kp_extractor
        self.generator = generator
        self.discriminator = discriminator
        self.train_params = train_params

        if sum(self.train_params['loss_weights']['perceptual']) != 0: 
            self.vgg = VGG19() 
            if torch.cuda.is_available(): 
                self.vgg = self.vgg.cuda() 

    def forward(self, x, use_both=False):

        loss_values = [] 
        
        approx_kp_joined = self.kp_extractor(torch.cat([x['source'], x['video']], dim=2))
        gt_kp_joined = self.kp_extractor(torch.cat([x['source'], x['gt_video']], dim=2))

        generated = self.generator(x['source'],
                                   **split_kp(approx_kp_joined, self.train_params['detach_kp_generator']))
        video_prediction = generated['video_prediction']
        video_deformed = generated['video_deformed']

        approx_kp_dict = split_kp(approx_kp_joined, self.train_params['detach_kp_generator']) 
        gt_kp_dict = split_kp(gt_kp_joined, self.train_params['detach_kp_generator']) 

        # Compute perceptual loss  
        if sum(self.train_params['loss_weights']['perceptual']) != 0: 
            perceptual_loss = 0.0 
            for scale in self.train_params['scales']:
                scaled_x = nn.AdaptiveAvgPool2d(int(scale*video_prediction.shape[-1]))(x['gt_video'].squeeze())
                scaled_y = nn.AdaptiveAvgPool2d(int(scale*video_prediction.shape[-1]))(video_prediction.squeeze())
                perceptual_loss += self.vgg(scaled_x, scaled_y, weights=self.train_params['loss_weights']['perceptual'])  
            loss_values.append(perceptual_loss)


        # For the first model, use the real ground truth key points (driving_B) so that 
        # the model hopefully retains the shape of src_B better, but could have worse 
        # aligned poses if driving_A does not match driving_B close enough. 
        
        # For the second model, 
        # we make the assumption that the shapes of the generated image correspond to the driving_A 
        # key points, and thus focus on those when comparing to the ground truth video. 
        
        for scale in self.train_params['scales']: 
            gt_scaled = nn.AdaptiveAvgPool2d(int(scale * video_prediction.shape[-1]))(x['gt_video'].squeeze()) \
                    .unsqueeze(2)
            video_pred_scaled = nn.AdaptiveAvgPool2d(int(scale * video_prediction.shape[-1]))(video_prediction.squeeze()) \
                    .unsqueeze(2)

            if not use_both: 
                discriminator_maps_generated = self.discriminator(video_pred_scaled, **gt_kp_dict)
                discriminator_maps_real = self.discriminator(gt_scaled, **gt_kp_dict)
                generated.update(gt_kp_dict)
            else:
                discriminator_maps_generated = self.discriminator(video_pred_scaled, **approx_kp_dict) 
                discriminator_maps_real = self.discriminator(gt_scaled, **gt_kp_dict) 
                generated['kp_driving'] = approx_kp_dict['kp_driving']
                generated['kp_source'] = approx_kp_dict['kp_source']
                generated['gt_kp_driving'] = gt_kp_dict['kp_driving'] 
           
            loss_values.extend(generator_loss(discriminator_maps_generated=discriminator_maps_generated,
                                discriminator_maps_real=discriminator_maps_real,
                                video_deformed=video_deformed,
                                loss_weights=self.train_params['loss_weights']))
        
        if use_both: 
            return tuple(loss_values) + (generated, approx_kp_joined, gt_kp_joined)
        else:
            return tuple(loss_values) + (generated, gt_kp_joined) 

class DiscriminatorFullModel(torch.nn.Module):
    """
    Merge all generator related updates into single model for better multi-gpu usage
    """

    def __init__(self, kp_extractor, generator, discriminator, train_params):
        super(DiscriminatorFullModel, self).__init__()
        self.kp_extractor = kp_extractor
        self.generator = generator
        self.discriminator = discriminator
        self.train_params = train_params

    def forward(self, x, kp_joined, generated, gt_kp_joined=None):

        kp_dict = split_kp(kp_joined, self.train_params['detach_kp_discriminator'])
 
        if gt_kp_joined is not None: 
            gt_kp_dict = split_kp(gt_kp_joined, self.train_params['detach_kp_discriminator']) 
            discriminator_maps_real = self.discriminator(x['gt_video'], **gt_kp_dict)
        else: 
            discriminator_maps_real = self.discriminator(x['gt_video'], **kp_dict) 
            
        discriminator_maps_generated = self.discriminator(generated['video_prediction'].detach(),
                **kp_dict)
        loss = discriminator_loss(discriminator_maps_generated=discriminator_maps_generated,
                                  discriminator_maps_real=discriminator_maps_real,
                                  loss_weights=self.train_params['loss_weights'])
        return loss

def compute_loss(generator, x, use_both=False):
    out = generator(x, use_both)
    if use_both: 
        loss_values = out[:-3]
        generated = out[-3]
        approx_kp_joined = out[-2]
        gt_kp_joined = out[-1]
        loss_values = [val.mean() for val in loss_values]
        loss = sum(loss_values)
        return loss_values, loss, generated, approx_kp_joined, gt_kp_joined 
    else: 
        loss_values = out[:-2] 
        generated = out[-2]
        kp_joined = out[-1] 
        loss_values = [val.mean() for val in loss_values]
        loss = sum(loss_values) 
        return loss_values, loss, generated, kp_joined 


def train(config, generator, discriminator, kp_detector, checkpoint, log_dir, dataset, device_ids,
        load_weights_only=False, use_both=False, update_kp=True):
    train_params = config['train_params']

    optimizer_generator = torch.optim.Adam(generator.parameters(), lr=train_params['lr'], betas=(0.5, 0.999))
    optimizer_discriminator = torch.optim.Adam(discriminator.parameters(), lr=train_params['lr'], betas=(0.5, 0.999))
    optimizer_kp_detector = torch.optim.Adam(kp_detector.parameters(), lr=train_params['lr'], betas=(0.5, 0.999))

    if checkpoint is not None:
        
        if load_weights_only:
            Logger.load_cpk(checkpoint, generator, discriminator, kp_detector)
            start_epoch = 0
            it = 0 
        else: 
            saved_start_epoch, saved_it =  Logger.load_cpk(checkpoint, generator, discriminator, 
                    kp_detector, optimizer_generator, optimizer_discriminator, optimizer_kp_detector)
            start_epoch = saved_start_epoch 
            it = saved_it 
    else:
        start_epoch = 0
        it = 0

    scheduler_generator = MultiStepLR(optimizer_generator, train_params['epoch_milestones'], gamma=0.1,
                                      last_epoch=start_epoch - 1)
    scheduler_discriminator = MultiStepLR(optimizer_discriminator, train_params['epoch_milestones'], gamma=0.1,
                                          last_epoch=start_epoch - 1)
    scheduler_kp_detector = MultiStepLR(optimizer_kp_detector, train_params['epoch_milestones'], gamma=0.1,
                                        last_epoch=start_epoch - 1)

    dataloader = DataLoader(dataset, batch_size=train_params['batch_size'], shuffle=True, num_workers=4, drop_last=True)

    generator_full = GeneratorFullModel(kp_detector, generator, discriminator, train_params)
    discriminator_full = DiscriminatorFullModel(kp_detector, generator, discriminator, train_params)

    generator_full_par = DataParallelWithCallback(generator_full, device_ids=device_ids)
    discriminator_full_par = DataParallelWithCallback(discriminator_full, device_ids=device_ids)


    if not os.path.isdir(log_dir + 'tb_log/'):
        os.mkdir(log_dir + 'tb_log/')

    writer = SummaryWriter(log_dir=log_dir + 'tb_log/') 


    with Logger(log_dir=log_dir, visualizer_params=config['visualizer_params'], **train_params['log_params']) as logger:
        for epoch in trange(start_epoch, train_params['num_epochs']):
            
            total_discriminator_loss = 0.
            total_generator_loss = 0.  

            for i, x in enumerate(dataloader):

                # driving = driving_A, source = src_B --> driving_B'
                mn1_dict = {} 
                mn1_dict['source'] = x['src_B']
                mn1_dict['video'] = x['driving_A']
                mn1_dict['gt_video'] = x['driving_B'] 

                '''This code is for the first model where we use ground truth key points for both''' 
                if not use_both:
                    loss_values_B, loss_B, generated_B, gt_kp_joined_B = compute_loss(generator_full_par,
                            mn1_dict)
                else:
                    '''This code is for the second model where we give GT and approx kp respectively '''
                    out_B = compute_loss(generator_full_par, mn1_dict, use_both=True) 
                    loss_values_B, loss_B, generated_B, approx_kp_joined_B, gt_kp_joined_B = out_B 

                # driving = generated_B (driving_B'), source = src_A --> driving_A'
                mn2_dict = {}
                mn2_dict['source'] = x['src_A'] 
                mn2_dict['video'] = generated_B['video_prediction']
                mn2_dict['gt_video'] = x['driving_A'] 
            
                # First model - see above 
                if not use_both: 
                    loss_values_A, loss_A, generated_A, gt_kp_joined_A = compute_loss(generator_full_par,
                            mn2_dict)
                else:
                    # Second model - see above  
                    out_A = compute_loss(generator_full_par, mn2_dict, use_both=True) 
                    loss_values_A, loss_A, generated_A, approx_kp_joined_A, gt_kp_joined_A = out_A 
            
                loss = loss_B + loss_A
                total_generator_loss += loss 
                
                loss = loss_B 
                total_generator_loss = loss 

                writer.add_scalar('generator loss B', loss_B.item(), it)
                writer.add_scalar('generator loss A', loss_A.item(), it) 
                writer.add_scalar('generator loss', loss.item(), it) 

                loss.backward(retain_graph=not train_params['detach_kp_discriminator'])
                optimizer_generator.step()
                optimizer_generator.zero_grad()
                optimizer_discriminator.zero_grad()

                if train_params['detach_kp_discriminator'] and update_kp:
                    optimizer_kp_detector.step()
                    optimizer_kp_detector.zero_grad()

                generator_loss_values = {}
                generator_loss_values['A'] = [val.detach().cpu().numpy() for val in loss_values_A]
                generator_loss_values['B'] = [val.detach().cpu().numpy() for val in loss_values_B] 


                if not use_both: 
                    loss_values_B = discriminator_full_par(mn1_dict, gt_kp_joined_B, generated_B)
                    loss_values_A = discriminator_full_par(mn2_dict, gt_kp_joined_A, generated_A) 
                else:
                    loss_values_B = discriminator_full_par(mn1_dict, approx_kp_joined_B, generated_B, 
                                                        gt_kp_joined_B)
                    loss_values_A = discriminator_full_par(mn2_dict, approx_kp_joined_A, generated_A, 
                                                        gt_kp_joined_A)
           
                loss_values_B = [val.mean() for val in loss_values_B]
                loss_values_A = [val.mean() for val in loss_values_A]
                
                loss_B = sum(loss_values_B)
                loss_A = sum(loss_values_A)

                loss = loss_A + loss_B 
                total_discriminator_loss += loss 
    
                loss = loss_B
                total_discriminator_loss = loss 

                writer.add_scalar('disc loss B', loss_B.item(), it)
                writer.add_scalar('disc loss A', loss_A.item(), it) 
                writer.add_scalar('disc loss', loss.item(), it) 

                loss.backward()
                optimizer_discriminator.step()
                optimizer_discriminator.zero_grad()
                if not train_params['detach_kp_discriminator'] and update_kp:
                    optimizer_kp_detector.step()
                    optimizer_kp_detector.zero_grad()

                discriminator_loss_values  = {}
                discriminator_loss_values['A'] = [val.detach().cpu().numpy() for val in loss_values_A]
                discriminator_loss_values['B'] = [val.detach().cpu().numpy() for val in loss_values_B]
                
                values = {
                        'A': generator_loss_values['A'] + discriminator_loss_values['A'],
                        'B': generator_loss_values['B'] + discriminator_loss_values['B'] 
                        }

                logger.log_iter(it,
                                names=generator_loss_names(train_params['loss_weights']) + 
                                discriminator_loss_names(),
                                values=values['B'], inp=mn1_dict, out=generated_B,
                                name='src_B_driving_A')
                logger.log_iter(it, 
                                names=generator_loss_names(train_params['loss_weights']) + 
                                discriminator_loss_names(), 
                                values=values['A'], inp=mn2_dict, out=generated_A,
                                name='src_A_driving_B')
                it += 1

            scheduler_generator.step()
            scheduler_discriminator.step()
            scheduler_kp_detector.step()

            writer.add_scalar('generator loss / train', total_generator_loss /(i + 1), epoch)
            writer.add_scalar('discriminator loss / train', total_discriminator_loss / (i + 1), epoch)

            logger.log_epoch(epoch, {'generator': generator,
                                     'discriminator': discriminator,
                                     'kp_detector': kp_detector,
                                     'optimizer_generator': optimizer_generator,
                                     'optimizer_discriminator': optimizer_discriminator,
                                     'optimizer_kp_detector': optimizer_kp_detector})
