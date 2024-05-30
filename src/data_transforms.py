### Transformation functions for fully supervised and DINO ###

from PIL import ImageOps, ImageFilter
import numpy as np
import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import random

IMG_MEAN = {'natural':[0.485, 0.456, 0.406], 
            'AS':[0.099, 0.099, 0.099],
            'TMED':[0.1122,0.1122,0.1122],
            'unit':[0.0, 0.0, 0.0]}
IMG_STD = {'natural':[0.229, 0.224, 0.225], 
           'AS':[0.171, 0.171, 0.171], 
           'TMED':[0.0321,0.0321,0.0321],
           'unit':[1.0, 1.0, 1.0]}

def get_deterministic_transform(img_resolution=224, mean=IMG_MEAN['natural'], std=IMG_STD['natural'], p_gamma=1.0):
    gamma_transform = AdaptiveGamma(p=p_gamma)
    tfd = transforms.Compose([
           transforms.Resize(size=(img_resolution, img_resolution)),
           gamma_transform,
           transforms.ToTensor(), 
           transforms.Normalize(mean, std)
           ])
    return tfd
    
def get_random_transform(img_resolution=224,  mean=IMG_MEAN['natural'], std=IMG_STD['natural'], p_gamma=0.5, min_crop_ratio=0.7, max_rotate_degrees=15):
    # Regarding what Tufts does: they don't go farther than saying "random crops and flips"
    gamma_transform = AdaptiveGamma(p=p_gamma)
    tfr = transforms.Compose([
           transforms.RandomResizedCrop(size=(img_resolution, img_resolution), scale=(min_crop_ratio, 1.0)),
           transforms.RandomRotation(degrees=max_rotate_degrees),
           gamma_transform,
           transforms.ToTensor(), 
           transforms.Normalize(mean, std)
           ])
    return tfr
    
def get_random_dino_transform(img_resolution=224,  mean=IMG_MEAN['natural'], std=IMG_STD['natural'], global_crops_scale=(0.4, 1)):
    return DinoTransform(img_resolution, mean, std, global_crops_scale)

class GaussianBlur(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if np.random.rand() < self.p:
            sigma = np.random.rand() * 1.4 + 0.1
            return img.filter(ImageFilter.GaussianBlur(sigma))
        else:
            return img
            
class IncreaseSharp(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if np.random.rand() < self.p:
            sharp = np.random.rand() * 3.9 + 1.1
            return TF.adjust_sharpness(img, sharp)
        else:
            return img

class AdaptiveGamma(object):
    def __init__(self, p):
        self.p = p
        
    def __call__(self, img):
        if np.random.rand() < self.p:
            gamma = np.log(0.5*255)/np.log(np.mean(img))
            return TF.adjust_gamma(img, 1/gamma)
        else:
            return img

class Solarization(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if np.random.rand() < self.p:
            return ImageOps.solarize(img)
        else:
            return img
            
# creds: https://github.com/facebookresearch/dino/blob/main/main_dino.py
class DinoTransform(object):
    def __init__(self, img_resolution, mean, std, global_crops_scale=(0.4, 1)):
        #, local_crops_scale=(0.25, 0.5), local_crops_number=0):
        assert img_resolution % 4 == 0
        flip_and_color_jitter = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.0)],
                p=0.8
            ),
            #transforms.RandomGrayscale(p=0.2),
            AdaptiveGamma(p=0.5),
        ])
        normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

        # first global crop
        self.global_transfo1 = transforms.Compose([
            transforms.RandomResizedCrop(img_resolution, scale=global_crops_scale, interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            utils.GaussianBlur(p=1.0),
            normalize,
        ])
        # second global crop
        self.global_transfo2 = transforms.Compose([
            transforms.RandomResizedCrop(img_resolution, scale=global_crops_scale, interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            utils.GaussianBlur(p=0.1),
            #utils.Solarization(0.2),
            normalize,
        ])
        # # transformation for the local small crops
        # self.local_crops_number = local_crops_number
        # self.local_transfo = transforms.Compose([
            # transforms.RandomResizedCrop(img_resolution//4, scale=local_crops_scale, interpolation=Image.BICUBIC),
            # flip_and_color_jitter,
            # #utils.GaussianBlur(p=0.5),
            # normalize,
        # ])

    def __call__(self, image):
        crops = []
        crops.append(self.global_transfo1(image))
        crops.append(self.global_transfo2(image))
        # # a multi-crop wrapper is used to handle images of different resolutions, where the images are placed together
        # for _ in range(self.local_crops_number):
            # crops.append(self.local_transfo(image))
        return crops