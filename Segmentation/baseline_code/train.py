import os
import random
import time
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import torchvision
import torchvision.transforms as transforms

import segmentation_models_pytorch as smp
from segmentation_models_pytorch.encoders import get_preprocessing_fn

import albumentations as A
from albumentations.pytorch import ToTensorV2

import matplotlib.pyplot as plt
import seaborn as sns; sns.set()

from importlib import import_module
import argparse

from utils import *
from load_data import *
from evaludate import *

# fix the seed
def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


# model save  
def save_model(model, saved_dir, file_name='best_model(pretrained).pt'):
    check_point = {'net': model.state_dict()}
    output_path = os.path.join(saved_dir, file_name)
    torch.save(model.state_dict(), output_path)


# fine-tuning
def fine_tuning(num_epochs, model, train_loader, val_loader, criterion, optimizer, saved_dir, device):
    print('-'*30)
    print('Start training..')

    best_mIoU = 0.0
    stop_count = 0
    
    for epoch in range(num_epochs):
        since = time.time()
        model.train()
        for step, (images, masks, _) in enumerate(train_loader):
            images = torch.stack(images)       # (batch, channel, height, width)
            masks = torch.stack(masks).long()  # (batch, channel, height, width)
            
            # gpu 연산을 위해 device 할당
            images, masks = images.to(device), masks.to(device)
                  
            # prediction
            outputs = model(images)
            
            # compute loss
            loss = criterion(outputs, masks)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # step 주기에 따른 loss 출력
            if (step + 1) % 25 == 0:
                step_time_elapsed = time.time() - since
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Time: {}m {}s'.format(
                    epoch+1, num_epochs, step+1, len(train_loader), loss.item(), step_time_elapsed // 60, step_time_elapsed % 60))
        
        #################### Validation ####################
        if (epoch + 1) % 1 == 0:            
            avrg_mIoU = validation(epoch + 1, model, val_loader, criterion, device)
            if avrg_mIoU > best_mIoU:
                best_mIoU = avrg_mIoU
                save_model(model, saved_dir)
                print('*****Best mIoU: {}*****'.format(best_mIoU))
                stop_count = 0
            else:
                stop_count += 1
        
        #################### Early Stopping ####################
        if stop_count >= 5:
            print('...Early Stopping...')
            print()
            break
        
        #################### Training Time(epoch) Check ####################
        time_elapsed = time.time() - since
        print('='*30)
        print(f'Training complete in {time_elapsed // 60}m {time_elapsed % 60}s')
        print('='*30)


# model train
def train(args):
    seed_everything(args.seed)
    device = torch.device('cuda:0')

    ########## Dataloader ##########
    train_transform = A.Compose([
        A.Resize(384, 384, p=1.0),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, p=1.0),
        A.ShiftScaleRotate(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomGridShuffle(p=0.5),
        A.MaskDropout(p=0.5),
        A.CenterCrop(256, 256, p=1.0),
        ToTensorV2(transpose_mask=True, p=1.0),
    ])

    val_transform = A.Compose([
        A.Resize(256, 256, p=1.0),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, p=1.0),
        ToTensorV2(transpose_mask=True, p=1.0),
    ])

    train_loader, val_loader = load_dataloader('../input/data', train_transform, val_transform, args.batch_size)

    if not os.path.isdir(args.saved_dir):                                                           
        os.mkdir(args.saved_dir)

    ########## Setting model ##########
    MODEL_NAME = getattr(import_module("segmentation_models_pytorch"), args.model_type)
    BACKBONE = args.backbone
    PRE_WEIGHT = args.pretrained_weight
    model = MODEL_NAME(
        encoder_name=BACKBONE, 
        encoder_weights=PRE_WEIGHT,
        in_channels=3,
        classes=12
    )
    model = model.to(device)
    
    ########## Loss function ##########
    loss_fn = getattr(import_module("torch.nn"), args.loss_fn)
    criterion = loss_fn()

    ########## Optimizer ##########
    optim = getattr(import_module("torch.optim"), args.optim)
    optimizer = optim(params=model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    ########## Model fine-tuning ##########
    fine_tuning(args.epochs, model, train_loader, val_loader, criterion, optimizer, args.saved_dir, device)


def main(args):
    train(args)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--model_type', type=str, default='Unet')
    parser.add_argument('--backbone', type=str, default='efficientnet-b0')
    parser.add_argument('--pretrained_weight', type=str, default='imagenet')
    parser.add_argument('--loss_fn', type=str, default='CrossEntropyLoss')
    parser.add_argument('--optim', type=str, default='AdamW')
    parser.add_argument('--weight_decay', type=float, default=1e-6)
    parser.add_argument('--saved_dir', type=str, default='./saved')
    

    args = parser.parse_args()
    print(args)

    main(args)
