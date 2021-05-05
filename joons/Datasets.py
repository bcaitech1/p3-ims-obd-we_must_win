import os

from torch.utils.data import Dataset
import cv2

import numpy as np

# 전처리를 위한 라이브러리
from pycocotools.coco import COCO
import torchvision
import torchvision.transforms as transforms

import albumentations as A
from albumentations.pytorch import ToTensorV2



# category_names = list(sorted_df.Categories)



class RecylceDatasets(Dataset):
    """COCO format"""
    def __init__(self, data_dir, mode = 'train', transform = None, resize=None):
        super().__init__()
        self.mode = mode
        self.transform = transform
        self.coco = COCO(data_dir)
        print('load_complete!')
        self.category_names = self.get_category_name()
        cat_ids = self.coco.getCatIds()
        self.cats = self.coco.loadCats(cat_ids)
        self.size = resize


    def __getitem__(self, index: int):
        # dataset이 index되어 list처럼 동작
        image_id = self.coco.getImgIds(imgIds=index)
        image_infos = self.coco.loadImgs(image_id)[0]

        # cv2 를 활용하여 image 불러오기
        images = cv2.imread(os.path.join('/opt/ml/input/data', image_infos['file_name']))
        images = cv2.cvtColor(images, cv2.COLOR_BGR2RGB)
        #.astype(np.float32)
        # images /= 255.0

        # Train, Test Loader
        if (self.mode in ('train', 'val')):
            # masks = self.get_annotation_data(image_infos)
            ann_ids = self.coco.getAnnIds(imgIds=image_infos['id'])
            anns = self.coco.loadAnns(ann_ids)

            # masks : size가 (height x width)인 2D
            # 각각의 pixel 값에는 "category id + 1" 할당
            # Background = 0
            masks = np.zeros((image_infos["height"], image_infos["width"]))
            # Unknown = 1, General trash = 2, ... , Cigarette = 11
            for ann in anns:
                pixel_value = ann['category_id'] + 1
                masks = np.maximum(self.coco.annToMask(ann) * pixel_value, masks)
            masks = masks.astype(np.float32)

            # transform -> albumentations 라이브러리 활용
            if self.size:
                resize = A.Compose([A.Resize(width=self.size, height=self.size)])
                transformed = resize(image=images, mask=masks)
                images = transformed["image"]
                masks = transformed["mask"]

            if self.transform is not None:
                transformed = self.transform(image=images, mask=masks)
                images = transformed["image"]
                masks = transformed["mask"]
            images = images/255.
            return images, masks, image_infos


        if self.mode == 'test':
            # transform -> albumentations 라이브러리 활용
            if self.size:
                resize = A.Compose([A.Resize(width=self.size, height=self.size)])
                transformed = resize(image=images)
                images = transformed["image"]
            if self.transform is not None:
                transformed = self.transform(image=images)
                images = transformed["image"]
            images = images / 255.
            return images, image_infos

    # category 이름형태의 list 출력하기
    def get_category_name(self):
        category_name = [0] * 11
        for c in self.coco.loadCats(self.coco.getCatIds()):
            category_name[c['id']] = c['name']
        category_name = ["Background"] + category_name
        return category_name



    def __len__(self) -> int:
        # 전체 dataset의 size를 return
        return len(self.coco.getImgIds())