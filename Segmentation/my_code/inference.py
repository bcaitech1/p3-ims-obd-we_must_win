import numpy as np
import pandas as pd

import torch

import segmentation_models_pytorch as smp
from segmentation_models_pytorch.encoders import get_preprocessing_fn

import albumentations as A
from albumentations.pytorch import ToTensorV2

from importlib import import_module
import argparse

from load_data import *

def test(model, test_loader, device):
    size = 256
    transform = A.Compose([A.Resize(256, 256)])
    print('-'*30)
    print('Start prediction.')
    model.eval()
    
    file_name_list = []
    preds_array = np.empty((0, size*size), dtype=np.long)
    
    with torch.no_grad():
        for step, (imgs, image_infos) in enumerate(test_loader):

            # inference (512 x 512)
            outs = model(torch.stack(imgs).to(device))
            oms = torch.argmax(outs, dim=1).detach().cpu().numpy()
            
            # resize (256 x 256)
            temp_mask = []
            for img, mask in zip(np.stack(imgs), oms):
                transformed = transform(image=img, mask=mask)
                mask = transformed['mask']
                temp_mask.append(mask)

            oms = np.array(temp_mask)
            
            oms = oms.reshape([oms.shape[0], size*size]).astype(int)
            preds_array = np.vstack((preds_array, oms))
            
            file_name_list.append([i['file_name'] for i in image_infos])
    print("End prediction.")
    print('-'*30)
    file_names = [y for x in file_name_list for y in x]
    
    return file_names, preds_array


def collate_fn(batch):
    return tuple(zip(*batch))


def load_test_data(dataset_path, test_transform, batch_size):
    test_path = dataset_path + '/test.json'

    # Dataset
    test_dataset = CustomDataLoader(data_dir=test_path, mode='test', transform=test_transform)

    # DataLoader
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        num_workers=4,
        collate_fn=collate_fn
    )

    return test_loader

def inference(args):
    device = torch.device('cuda:0')

    ########## best model 불러오기 ##########
    model_path = './saved/best_model(pretrained).pt'

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

    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint)

    # dataset
    test_transform = A.Compose([
        A.Resize(256, 256, p=1.0),
        # A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, p=1.0),
        ToTensorV2(),       
    ])
    test_loader = load_test_data('../input/data', test_transform, args.batch_size)

    # inference
    return test(model, test_loader, device)


def main(args):
    submission = pd.read_csv('./submission/sample_submission.csv', index_col=None)
    file_names, preds = inference(args)

    # PredictionString 대입
    for file_name, string in zip(file_names, preds):
        submission = submission.append({"image_id" : file_name, "PredictionString" : ' '.join(str(e) for e in string.tolist())}, ignore_index=True)

    # submission.csv로 저장
    submission.to_csv(args.out_path, index=False)
    print('Done!')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--out_path', type=str, default="./submission/submission.csv")
    parser.add_argument('--model_type', type=str, default='DeepLabV3Plus')
    parser.add_argument('--backbone', type=str, default='resnext50_32x4d')
    parser.add_argument('--pretrained_weight', type=str, default='swsl')
    
    args = parser.parse_args()
    print(args)

    main(args)