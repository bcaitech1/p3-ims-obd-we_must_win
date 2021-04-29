from torch.utils.data import Dataset, DataLoader
from pycocotools.coco import COCO
from utils import make_class_dict
import os
import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2


class MyDataset(Dataset):
    def __init__(self, file_name, data_path="/opt/ml/input/data", class_dict=None, transform=None):
        super().__init__()
        self.data_path = data_path
        self.transform = transform
        self.class_dict = class_dict
        self.coco = COCO(os.path.join(self.data_path, file_name))
        self.category_id_list = self.coco.getCatIds()
        self.category_list = self.coco.loadCats(self.category_id_list)

    def __len__(self):
        return len(self.coco.getImgIds())

    @staticmethod
    def getClassname(class_id, category_list):
        for i in range(len(category_list)):
            if category_list[i]['id'] == class_id:
                return category_list[i]['name']
        return "Background"

    def __getitem__(self, idx):
        image_id = self.coco.getImgIds(imgIds=idx)
        image_infos = self.coco.loadImgs(image_id)[0]

        image = cv2.imread(os.path.join(self.data_path, image_infos["file_name"]))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)

        if not self.class_dict:
            if self.transform:
                trans = self.transform(image=image)
                image = trans["image"]
            return image, image_infos
        annotation_id_list = self.coco.getAnnIds(imgIds=image_infos['id'])
        annotation_list = self.coco.loadAnns(annotation_id_list)
        mask = np.zeros((image_infos["height"], image_infos["width"]))
        for i in range(len(annotation_list)):
            class_name = self.getClassname(annotation_list[i]['category_id'], self.category_list)
            value = self.class_dict[class_name]
            mask = np.maximum(self.coco.annToMask(annotation_list[i]) * value, mask)
        mask = mask.astype(np.float32)
        if self.transform:
            trans = self.transform(image=image, mask=mask)
            image = trans["image"]
            mask = trans["mask"]

        return image, mask


def get_loader(batch_size, trans_option, width, height):
    class_dict = make_class_dict()
    val_transform = A.Compose([A.Resize(width, height),
                               A.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                               ToTensorV2()])
    train_transform = get_trans(trans_option, width, height)
    val_dataset = MyDataset(file_name="val.json", class_dict=class_dict, transform=val_transform)
    train_dataset = MyDataset(file_name="train.json", class_dict=class_dict, transform=train_transform)

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    return train_loader, val_loader


def get_trans(option, width, height):
    if option == 0:
        return A.Compose([A.Resize(width, height),
                          A.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                          ToTensorV2()])
    if option == 1:
        return A.Compose([A.Resize(width, height),
                          A.RandomGridShuffle(),
                          A.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                          ToTensorV2()])
    if option == 2:
        return A.Compose([A.Resize(width, height),
                          A.Rotate(),
                          A.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                          ToTensorV2()])
    if option == 3:
        return A.Compose([A.Resize(width, height),
                          A.MaskDropout(),
                          A.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                          ToTensorV2()])
    if option == 4:
        return A.Compose([A.Resize(width, height),
                          A.GridDropout(),
                          A.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                          ToTensorV2()])
    if option == 5:
        return A.Compose([A.Resize(width, height),
                          A.GridDistortion(),
                          A.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                          ToTensorV2()])
    if option == 6:
        return A.Compose([A.Resize(width, height),
                          A.GaussNoise(),
                          A.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                          ToTensorV2()])
    if option == 7:
        return A.Compose([A.Resize(width, height),
                          A.RandomBrightnessContrast(),
                          A.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                          ToTensorV2()])
    if option == 8:
        return A.Compose([A.Resize(width, height),
                          A.HorizontalFlip(),
                          A.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                          ToTensorV2()])


