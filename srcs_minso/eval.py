from dataset import MyDataset
from torch.utils.data import DataLoader
import albumentations as A
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from model import get_model
import pandas as pd
import os


def collate_fn(batch):
    return tuple(zip(*batch))


def base_test(args, device, test_loader, size, resize):
    if args.use_lib:
        model = get_model(args.architecture, args.backbone, args.backbone_weight).to(device)
    else:
        # SWIM 구현 후 변경 예정
        pass
    model.load_state_dict(torch.load(os.path.join(args.save_dir, "best_miou.pt")))
    print('Start prediction.')
    model.eval()

    file_names = []
    preds = np.empty((0, size * size), dtype=np.long)
    with torch.no_grad():
        for step, (imgs, image_infos) in enumerate(test_loader):
            outs = model(torch.stack(imgs).to(device))
            oms = torch.argmax(outs, dim=1).detach().cpu().numpy()
            if oms.shape[1] != 256 or oms.shape[2] != 256:
                transformed = resize(image=torch.stack(imgs).squeeze(), mask=oms.squeeze())
                oms = np.array(transformed["mask"])
            oms = oms.reshape([oms.shape[0], size * size]).astype(int)
            preds = np.vstack((preds, oms))
            file_names.append(image_infos[0]["file_name"])
    return file_names, preds


def fold_ensemble(args, device, test_loader, size, resize):
    models = []
    for i in range(5):
        if args.use_lib:
            model = get_model(args.architecture, args.backbone, args.backbone_weight).to(device)
        else:
            # SWIM 구현 후 변경 예정
            pass
        model.load_state_dict(torch.load(os.path.join(args.save_dir, f"{i}best_miou.pt")))
        model.eval()
        models.append(model)
    file_names = []
    preds = np.empty((0, size * size), dtype=np.long)
    with torch.no_grad():
        for step, (imgs, image_infos) in enumerate(test_loader):
            for i in range(len(models)):
                if i == 0:
                    oms = models[i](torch.stack(imgs).to(device))
                else:
                    oms += models[i](torch.stack(imgs).to(device))
            oms = torch.argmax(oms, dim=1).detach().cpu().numpy()
            if oms.shape[1] != 256 or oms.shape[2] != 256:
                transformed = resize(image=torch.stack(imgs).squeeze(), mask=oms.squeeze())
                oms = np.array(transformed["mask"])
            oms = oms.reshape([oms.shape[0], size * size]).astype(int)
            preds = np.vstack((preds, oms))
            file_names.append(image_infos[0]["file_name"])
    return file_names, preds


def inference(args, device):
    test_transform = A.Compose([
        A.Resize(args.width, args.height),
        A.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
    test_dataset = MyDataset(file_name="test.json", transform=test_transform)
    test_loader = DataLoader(dataset=test_dataset, batch_size=1, num_workers=4, collate_fn=collate_fn)
    size = 256
    resize = A.Compose([A.Resize(256, 256)])

    print('Start prediction.')
    if args.kfold == 1 or args.kfold == 2:
        file_names, preds = fold_ensemble(args, device, test_loader, size, resize)
    else:
        file_names, preds = base_test(args, device, test_loader, size, resize)

    submission = pd.read_csv('/opt/ml/code/submission/sample_submission.csv', index_col=None)
    for file_name, string in zip(file_names, preds):
        submission = submission.append(
            {"image_id": file_name, "PredictionString": ' '.join(str(e) for e in string.tolist())},
            ignore_index=True)
    submission.to_csv(f"/opt/ml/code/submission/{args.id}.csv", index=False)
    print("End prediction.")
