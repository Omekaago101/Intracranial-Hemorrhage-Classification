import cv2
import os
import torch
import numpy as np
from torch.utils.data import Dataset
from albumentations.pytorch.transforms import ToTensorV2
from albumentations import (MedianBlur, Compose, Normalize, OpticalDistortion, HorizontalFlip,
                           VerticalFlip, ShiftScaleRotate, Transpose, OneOf, IAAAdditiveGaussianNoise,
                           GaussNoise, RandomGamma, Blur, RandomBrightness, HueSaturationValue,
                           RandomBrightnessContrast, GridDistortion,Lambda, NoOp, CenterCrop, Resize,RandomResizedCrop
                           )
class IntracranialDataset(Dataset):
    
    def __init__(self, cfg, df, path, labels,AUTOCROP,HFLIP,TRANSPOSE,mode='train'):
        self.path = path
        self.data = df
        self.labels = labels
        self.crop = AUTOCROP
        self.cfg = cfg
        self.mode = mode
        self.transpose = TRANSPOSE
        self.hflip = HFLIP
        self.lbls = cfg.CONST.LABELS
        if self.mode == "train":
            self.transform = Compose([
                RandomResizedCrop(cfg.DATA.IMG_SIZE, cfg.DATA.IMG_SIZE,
                                    interpolation=cv2.INTER_LINEAR, scale=(0.8, 1)),
                OneOf([
                    HorizontalFlip(p=1.),
                    VerticalFlip(p=1.),
                ]),
                OneOf([
                    ShiftScaleRotate(
                        shift_limit=0.0625,
                        scale_limit=0.1,
                        rotate_limit=30,
                        border_mode=cv2.BORDER_CONSTANT,
                        value=0,
                        p=1.),
                    GridDistortion(
                        distort_limit=0.2,
                        border_mode=cv2.BORDER_CONSTANT,
                        value=0,
                        p=1.),
                    OpticalDistortion(
                        distort_limit=0.2,
                        shift_limit=0.15,
                        border_mode=cv2.BORDER_CONSTANT,
                        value=0,
                        p=1.),
                    NoOp()
                ]),
                OneOf([
                    IAAAdditiveGaussianNoise(p=1.),
                    GaussNoise(p=1.),
                    NoOp()
                ]),
                OneOf([
                    MedianBlur(blur_limit=3, p=1.),
                    Blur(blur_limit=3, p=1.),
                    NoOp()
                ])
            ])
        elif self.mode == 'test' or self.mode == 'valid':
            HFLIPVAL = 1.0 if self.hflip == 'T' else 0.0
            TRANSPOSEVAL = 1.0 if self.transpose == 'P' else 0.0
            self.transform = Compose([
                HorizontalFlip(p=HFLIPVAL),
                Transpose(p=TRANSPOSEVAL),
                Normalize(mean=[0.22363983, 0.18190407, 0.2523437 ], 
                          std=[0.32451536, 0.2956294,  0.31335256], max_pixel_value=255.0, p=1.0),
            ])
        self.totensor = ToTensorV2()
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = os.path.join(self.path, self.data.loc[idx, 'Image'] + '.jpg')
        #img = cv2.imread(img_name, cv2.IMREAD_GRAYSCALE)   
        img = cv2.imread(img_name)    
        if self.crop:
            try:
                try:
                    img = self.autocrop(img, threshold=0, kernsel_size = img.shape[0]//15)
                except:
                    img = self.autocrop(img, threshold=0)  
            except:
                1  
        img = cv2.resize(img,(self.cfg.DATA.IMG_SIZE,self.cfg.DATA.IMG_SIZE))
        if self.mode == "train":       
            augmented = self.transform(image=img)
            img = augmented['image']   
        if self.labels:
            labels = torch.tensor(
                self.data.loc[idx, self.cfg.CONST.LABELS])
            return {'image': img, 'labels': labels}    
        else:      
            return {'image': img}
    
    def autocrop(image, threshold=0):
        """Crops any edges below or equal to threshold
        Crops blank image to 1x1.
        Returns cropped image.
        https://stackoverflow.com/questions/13538748/crop-black-edges-with-opencv
        """

        if len(image.shape) == 3:
            flatImage = np.max(image, 2)
        else:
            flatImage = image
        rows = np.where(np.max(flatImage, 0) > threshold)[0]
        cols = np.where(np.max(flatImage, 1) > threshold)[0]
        image = image[cols[0]: cols[-1] + 1, rows[0]: rows[-1] + 1]
        #logger.info(image.shape)
        sqside = max(image.shape)
        imageout = np.zeros((sqside, sqside, 3), dtype = 'uint8')
        imageout[:image.shape[0], :image.shape[1],:] = image.copy()
        return imageout