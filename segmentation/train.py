import os
import json
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import albumentations as A
from albumentations.pytorch import ToTensorV2

import segmentation_models_pytorch as smp

# ======= custom module ======= #
import models
from inference import test
import utils

from trainer import train
from data import CustomDataLoader



print('pytorch version: {}'.format(torch.__version__))
print('GPU 사용 가능 여부: {}'.format(torch.cuda.is_available()))
print(torch.cuda.get_device_name(0))
print(torch.cuda.device_count())
device = "cuda" if torch.cuda.is_available() else "cpu"   # GPU 사용 가능 여부에 따라 device 정보 저장

utils.seed_everything(seed=42)


dataset_path = "/opt/ml/input/data"
train_path = dataset_path + '/train.json'
val_path = dataset_path + '/val.json'
test_path = dataset_path + '/test.json'

# ======= 파라미터 ======= #
val_every = 1 
batch_size = 16
num_epochs = 30
learning_rate = 0.0001
patience = 5

# ======= 데이터셋, 데이터로더 구성 ======= #
def collate_fn(batch):
    return tuple(zip(*batch))
# normalize, maskdropout, 
train_transform = A.Compose([A.Resize(256,256), ToTensorV2()])
val_transform = A.Compose([A.Resize(256,256), ToTensorV2()])
test_transform = A.Compose([A.Resize(256,256), ToTensorV2()])

train_dataset = CustomDataLoader(data_dir=train_path, img_dir=dataset_path, mode='train', transform=train_transform)
val_dataset = CustomDataLoader(data_dir=val_path, img_dir=dataset_path, mode='val', transform=val_transform)
test_dataset = CustomDataLoader(data_dir=test_path, img_dir=dataset_path, mode='test', transform=test_transform)


train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                           batch_size=batch_size,
                                           shuffle=True,
                                           num_workers=4,
                                           collate_fn=collate_fn)

val_loader = torch.utils.data.DataLoader(dataset=val_dataset, 
                                         batch_size=batch_size,
                                         shuffle=False,
                                         num_workers=4,
                                         collate_fn=collate_fn)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False,
                                          num_workers=4,
                                          collate_fn=collate_fn)



# ======= 모델 구성 ======= #
architecture = "DeepLabV3+"
backbone = "resnext50_32x4d"
weight = "swsl" # imagenet / ssl / swsl

model = models.get_model(architecture=architecture, backbone=backbone, weight=weight).to(device)

criterion = nn.CrossEntropyLoss()


optimizer = torch.optim.AdamW(params = model.parameters(), lr = learning_rate, weight_decay=1e-6)
model_name = architecture + "_" + weight +  "_best_model(pretrained)" + ".pt"
model_path = '../saved/' + model_name

# epoch 30, early stop 5, miou 기준 모델 저장
best_mIoU = train(num_epochs=num_epochs,
                  model=model,
                  model_name=model_name,
                  train_loader=train_loader,
                  val_loader=val_loader,
                  criterion=criterion, 
                  optimizer=optimizer, 
                  patience=patience, 
                  val_every=val_every, 
                  device=device)


# best model 불러오기
checkpoint = torch.load(model_path, map_location=device)
model.load_state_dict(checkpoint)



# sample_submisson.csv 열기
submission = pd.read_csv('/opt/ml/code/submission/sample_submission.csv', index_col=None)
file_names, preds = test(model, test_loader, device, print_every=25)

# PredictionString 대입
for file_name, string in zip(file_names, preds):
    submission = submission.append({"image_id" : file_name, "PredictionString" : ' '.join(str(e) for e in string.tolist())}, 
                                   ignore_index=True)

# submission.csv로 저장
submission.to_csv("/opt/ml/code/submission/" + architecture + "_" + weight + "_" + str(best_mIoU)[:6] + ".csv", index=False)
print('submission saved!')