import os
import json

import numpy as np
import pandas as pd
import torch

import cv2
import albumentations as A

from torch.utils.data import DataLoader

from dataset import CocoDataset
from build import Model

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def collate_fn(batch):
    return tuple(zip(*batch))


def inference():
    output_id = "PST-53"
    score_type = "mIoU"

    save_path = os.path.join("outputs", output_id)
    with open(os.path.join(save_path, "configs.json"), "r") as config_f:
        CONFIGS = json.load(config_f)

    data_path = "./input/data"
    test_path = os.path.join(data_path, "test.json")
    test_dataset = CocoDataset(
        json_path=test_path,
        data_path=data_path,
        input_size=CONFIGS["INPUT_SIZE"],
        norm_mean=CONFIGS["NORM_MEAN"],
        norm_std=CONFIGS["NORM_STD"],
        mode="test",
    )
    test_loader = DataLoader(dataset=test_dataset, batch_size=CONFIGS["BATCH_SIZE"], num_workers=4, collate_fn=collate_fn)

    model = Model(model_name=CONFIGS["MODEL_NAME"], **CONFIGS["MODEL_PARAMS"]).to(device)
    model.load_state_dict(torch.load(os.path.join(save_path, f"best_{score_type}_model.pth")))
    model.eval()

    file_names = []
    preds_array = np.empty((0, 256 * 256), dtype=np.long)

    size = 256
    transform = A.Compose([A.Resize(size, size, interpolation=cv2.INTER_AREA, always_apply=True)])

    def mask_transform(image, mask):
        image = image.transpose(1, 2, 0)
        if (*image.shape[:2],) != (*(size, size),):
            transformed = transform(image=image, mask=mask)
            mask = transformed["mask"]
        return mask

    print("Test data inference starts ...")
    with torch.no_grad():
        for step, (images, image_infos) in enumerate(test_loader, 1):
            images = torch.stack(images).to(device)

            outputs = model(images)

            pred_mask = torch.argmax(outputs, dim=1).detach().cpu().numpy()
            pred_mask = [mask_transform(image, mask) for image, mask in zip(np.stack(images.detach().cpu().numpy()), pred_mask)]
            pred_mask = np.array(pred_mask)
            pred_mask = pred_mask.reshape([pred_mask.shape[0], size * size]).astype(int)

            preds_array = np.vstack((preds_array, pred_mask))
            file_names.extend([i["file_name"] for i in image_infos])
            print(f"inference iteration - {step}/{len(test_loader)}", end="\r")
    print()

    print("Creating submission csv file ... ")
    submission = pd.read_csv("sample_submission.csv", index_col=None)
    submit_data = zip(file_names, preds_array)
    for step, (file_name, ans) in enumerate(submit_data, 1):
        submission = submission.append({"image_id": file_name, "PredictionString": " ".join(str(e) for e in ans.tolist())}, ignore_index=True)
        print(f"creating csv iteration - {step}/{len(file_names)}", end="\r")
    print()
    submission.to_csv(os.path.join(save_path, f"{output_id}_best_{score_type}.csv"), index=False)
    print("All done !")


if __name__ == "__main__":
    inference()
