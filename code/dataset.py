import os

import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2

from torch.utils.data import Dataset
from pycocotools.coco import COCO


transform2tensor = ToTensorV2()

category_names = [
    "Background",
    "UNKNOWN",
    "General trash",
    "Paper",
    "Paper pack",
    "Metal",
    "Glass",
    "Plastic",
    "Styrofoam",
    "Plastic bag",
    "Battery",
    "Clothing",
]


class CocoDataset(Dataset):
    def __init__(self, json_path, data_path, input_size=(512, 512), norm_mean=(0, 0, 0), norm_std=(1, 1, 1), mode="train", transform=None):
        super().__init__()
        self.input_size = input_size
        self.normalizer = A.Compose([A.Normalize(mean=norm_mean, std=norm_std, always_apply=True)])
        self.resizer = A.Compose([A.Resize(*input_size, interpolation=cv2.INTER_AREA, always_apply=True)])
        self.mode = mode
        self.transform = transform

        self.coco = COCO(json_path)
        self.data_path = data_path

    def __getitem__(self, idx):
        # multiple indexing and slicing.
        if hasattr(idx, "__iter__") or isinstance(idx, slice):
            if isinstance(idx, slice):
                idx = range(*idx.indices(len(self)))

            if self.mode in ["train", "valid"]:
                images, masks, image_infos = [], [], []
                for i in idx:
                    image, mask, image_info = self.__getitem__(i)
                    images.append(image)
                    masks.append(mask)
                    image_infos.append(image_info)

                return images, masks, image_infos

            if self.mode == "test":
                images, image_infos = [], []
                for i in idx:
                    image, image_info = self.__getitem__(i)
                    images.append(image)
                    image_infos.append(image_info)

                return images, image_infos

            raise Exception("Mode argument must be one of [train, valid, test].")

        # single indexing.
        else:
            image_id = self.coco.getImgIds(imgIds=idx)
            image_infos = self.coco.loadImgs(image_id)[0]

            image = cv2.imread(os.path.join(self.data_path, image_infos["file_name"]))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            if self.mode in ["train", "valid"]:
                ann_ids = self.coco.getAnnIds(imgIds=image_infos["id"])
                anns = self.coco.loadAnns(ann_ids)

                cat_ids = self.coco.getCatIds()
                cats = self.coco.loadCats(cat_ids)

                mask = np.zeros((image_infos["height"], image_infos["width"]))
                for ann in anns:
                    class_name = get_classname(ann["category_id"], cats)
                    pixel_value = category_names.index(class_name)
                    mask = np.maximum(self.coco.annToMask(ann) * pixel_value, mask)

                if self.transform is not None:
                    transformed = self.transform(image=image, mask=mask)
                    image = transformed["image"]
                    mask = transformed["mask"]

                # Resize to input size.
                if (*image.shape[:2],) != (*self.input_size,):
                    transformed = self.resizer(image=image, mask=mask)
                    image = transformed["image"]
                    mask = transformed["mask"]

                # Normalize image.
                transformed = self.normalizer(image=image, mask=mask)
                image = transformed["image"]
                mask = transformed["mask"]

                image = image.astype(np.float32)
                mask = mask.astype(np.long)

                # Convert to tensor.
                transformed = transform2tensor(image=image, mask=mask)
                image = transformed["image"]
                mask = transformed["mask"]

                return image, mask, image_infos

            if self.mode == "test":
                if self.transform is not None:
                    transformed = self.transform(image=image)
                    image = transformed["image"]

                # Resize to input size.
                if (*image.shape[:2],) != (*self.input_size,):
                    transformed = self.resizer(image=image)
                    image = transformed["image"]

                # Normalize image.
                transformed = self.normalizer(image=image)
                image = transformed["image"]

                # Convert to tensor.
                transformed = transform2tensor(image=image)
                image = transformed["image"]

                return image, image_infos

            raise Exception("Mode argument must be one of [train, valid, test].")

    def __len__(self):
        return len(self.coco.getImgIds())


def get_classname(classID, cats):
    for cat in cats:
        if cat["id"] == classID:
            return cat["name"]
    return "None"


# sanity check.
if __name__ == "__main__":
    json_path = "input/data/train.json"
    data_path = "input/data"
    train_dataset = CocoDataset(json_path, data_path, input_size=(256, 256), mode="train", transform=None)
    print(train_dataset[0])
