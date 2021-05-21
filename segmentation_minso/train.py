import torch.optim as optim
import torch.nn as nn
import torch
from validation import validation
from dataset import get_loader
from model import get_model
from utils import get_criterion, get_scheduler, get_optimizer, get_dataframe
from utils import EarlyStopping
import time
import datetime
import os
from adamp import AdamP
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split


def train(args, device):
    best_miou = 0
    if args.use_lib:
        model = get_model(args.architecture, args.backbone, args.backbone_weight).to(device)
    else:
        # SWIM 구현 후 변경 예정
        model = get_model(args.architecture, args.backbone, args.backbone_weight).to(device)
    data_loader, val_loader = get_loader(args.batch_size, args.transform, args.width, args.height)
    optimizer = get_optimizer(model, args.optimizer, args.lr)
    scheduler = get_scheduler(args.scheduler, optimizer)
    criterion = get_criterion()
    early_stop = EarlyStopping()
    miou_log = []
    if args.scheduler == 0:
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
    if args.scheduler == 1:
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-6)
    if args.scheduler == 2:
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=100, T_mult=2, eta_min=1e-6)
    for epoch in range(args.epochs):
        model.train()
        for step, (images, masks) in enumerate(data_loader):
            masks = masks.long()
            images, masks = images.to(device), masks.to(device)
            pred = model(images)
            loss = criterion(pred, masks)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if args.scheduler == 1 or args.scheduler == 2:
                scheduler.step()
        val_loss, miou = validation(model, val_loader, criterion, device)
        miou_log.append(miou)
        print(f"#{epoch + 1}/{args.epochs}  loss : {val_loss:.4}  mIoU : {miou:.4}")
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), os.path.join(args.save_dir, "best_loss.pt"))
        if args.early_stop and early_stop.validate(val_loss, miou):
            break
        if args.scheduler == 0:
            scheduler.step()
    print(miou_log)
    return best_miou


def kfold_train(model, criterion, optimizer, device, early_stop, args):
    best_loss = 99999
    best_miou = 0

    return best_miou, best_loss


def stratified_kfold_train(args, device):
    data_info = get_dataframe()
    kf = StratifiedKFold(n_splits=5, shuffle=True)
    early_stop = EarlyStopping()

    for fold, (train_index, valid_index) in enumerate(kf.split(data_info, list(data_info.loc[:, "representation_label"]))):
        early_stop.clear()
        best_miou = 0
        if args.use_lib:
            model = get_model(args.architecture, args.backbone, args.backbone_weight).to(device)
        else:
            # SWIM 구현 후 변경 예정
            model = get_model(args.architecture, args.backbone, args.backbone_weight).to(device)
        train_info = data_info.loc[train_index, :]
        valid_info = data_info.loc[valid_index, :]
        data_loader, val_loader = get_loader(args.batch_size, args.transform, args.width, args.height, train_info, valid_info, True)
        optimizer = get_optimizer(model, args.optimizer, args.lr)
        scheduler = get_scheduler(args.scheduler, optimizer)
        criterion = get_criterion()
        miou_log = []
        for epoch in range(args.epochs):
            model.train()
            for step, (images, masks) in enumerate(data_loader):
                masks = masks.long()
                images, masks = images.to(device), masks.to(device)
                pred = model(images)
                loss = criterion(pred, masks)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if args.scheduler == 1 or args.scheduler == 2:
                    scheduler.step()
            val_loss, miou = validation(model, val_loader, criterion, device)
            miou_log.append(miou)
            print(f"#{fold} {epoch + 1}/{args.epochs}  loss : {val_loss:.4}  mIoU : {miou:.4}")
            if miou > best_miou:
                best_miou = miou
                torch.save(model.state_dict(), os.path.join(args.save_dir, f"{fold}best_miou.pt"))
            if args.early_stop and early_stop.validate(val_loss, miou):
                break
            if args.scheduler == 0:
                scheduler.step()
        print(miou_log)
    return best_miou


def trainer(args, device):
    print(f"Start train")
    start = time.time()
    if args.kfold == 1:
        best_iou = kfold_train(args, device)
    elif args.kfold == 2:
        best_iou = stratified_kfold_train(args, device)
    else:
        best_iou = train(args, device)
    sec = time.time() - start
    times = str(datetime.timedelta(seconds=sec)).split(".")
    print(f"End train {times[0]}")
    return best_iou
