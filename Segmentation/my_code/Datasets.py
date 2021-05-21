import os
import glob

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
            return images, masks


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




def get_classname(classID, cats):
    for i in range(len(cats)):
        if cats[i]['id']==classID:
            return cats[i]['name']
    return "None"

class PseudoTrainset(Dataset):
    """COCO format"""

    def __init__(self, data_dir, transform=None, resize=None):
        super().__init__()
        self.size = resize
        self.transform = transform
        self.coco = COCO(data_dir)
        self.dataset_path = '/opt/ml/input/data/'
        self.category_names = ['Backgroud', 'UNKNOWN', 'General trash', 'Paper', 'Paper pack', 'Metal', 'Glass',
                               'Plastic', 'Styrofoam', 'Plastic bag', 'Battery', 'Clothing']

        self.pseudo_imgs = np.load(self.dataset_path + 'pseudo_imgs_path.npy')
        self.pseudo_masks = sorted(glob.glob(self.dataset_path + 'pseudo_masks/*.npy'))

    def __getitem__(self, index: int):

        ### Train data ###
        if (index < len(self.coco.getImgIds())):
            image_id = self.coco.getImgIds(imgIds=index)
            image_infos = self.coco.loadImgs(image_id)[0]

            images = cv2.imread(self.dataset_path + image_infos['file_name'])
            images = cv2.cvtColor(images, cv2.COLOR_BGR2RGB)\
            # .astype(np.float32)
            # images /= 255.0
            ann_ids = self.coco.getAnnIds(imgIds=image_infos['id'])
            anns = self.coco.loadAnns(ann_ids)
            cat_ids = self.coco.getCatIds()
            cats = self.coco.loadCats(cat_ids)

            ###  mask 생성  ###
            masks = np.zeros((image_infos["height"], image_infos["width"]))
            for i in range(len(anns)):
                className = get_classname(anns[i]['category_id'], cats)
                pixel_value = self.category_names.index(className)
                masks = np.maximum(self.coco.annToMask(anns[i]) * pixel_value, masks)

        ### Pseudo data ###
        else:
            index -= len(self.coco.getImgIds())
            images = cv2.imread(self.dataset_path + self.pseudo_imgs[index])
            images = cv2.cvtColor(images, cv2.COLOR_BGR2RGB)\
            # .astype(np.float32)
            # images /= 255.0
            masks = np.load(self.pseudo_masks[index])

        if self.size:
            resize = A.Compose([A.Resize(width=self.size, height=self.size)])
            transformed = resize(image=images, mask=masks)
            images = transformed["image"]
            masks = transformed["mask"]

        ###  augmentation ###
        # masks = masks.astype(np.float32)
        if self.transform is not None:
            transformed = self.transform(image=images, mask=masks)
            images = transformed["image"]
            masks = transformed["mask"]
        images = images/255.
        return images, masks

    def __len__(self):
        return len(self.coco.getImgIds()) + len(self.pseudo_imgs)