import torch.optim as optim
import torch.nn as nn
import torch
from validation import validation
from dataset import get_loader
from utils import EarlyStopping
import time
import datetime
import os


def train(epochs, model, data_loader, val_loader, criterion, optimizer, device, early_stop, save_dir):
    best_loss = 99999
    best_miou = 0
    for epoch in range(epochs):
        model.train()
        for step, (images, masks) in enumerate(data_loader):
            masks = masks.long()
            images, masks = images.to(device), masks.to(device)
            pred = model(images)
            loss = criterion(pred, masks)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        val_loss, miou = validation(model, val_loader, criterion, device)
        print(f"#{epoch + 1}/{epochs}  loss : {val_loss:.4}  mIoU : {miou:.4}")
        if miou > best_miou:
            best_miou = miou
            torch.save(model.state_dict(), os.path.join(save_dir, "best_miou.pt"))
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), os.path.join(save_dir, "best_loss.pt"))
        if early_stop.validate(val_loss, miou):
            break
    return best_miou, best_loss


def kfold_train(epochs, model, data_loader, val_loader, criterion, optimizer, device, early_stop, save_dir):
    best_loss = 99999
    best_miou = 0

    return best_miou, best_loss


def stratified_kfold_train(epochs, model, data_loader, val_loader, criterion, optimizer, device, early_stop, save_dir):
    best_loss = 99999
    best_miou = 0

    return best_miou, best_loss


def trainer(args, model, device):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(params=model.parameters(), lr=args.lr)
    # optimizer = optim.Adam(params=model.parameters(), lr=args.lr)
    train_loader, val_loader = get_loader(args.batch_size, args.transform, args.width, args.height)
    early_stop = EarlyStopping()
    print(f"Start train")
    start = time.time()
    if args.kfold == 1:
        best_iou, best_loss = kfold_train(args.epochs, model, train_loader, val_loader, criterion, optimizer, device, early_stop, args.save_dir)
    elif args.kfold == 2:
        best_iou, best_loss = stratified_kfold_train(args.epochs, model, train_loader, val_loader, criterion, optimizer, device, early_stop, args.save_dir)
    else:
        best_iou, best_loss = train(args.epochs, model, train_loader, val_loader, criterion, optimizer, device, early_stop, args.save_dir)
    sec = time.time() - start
    times = str(datetime.timedelta(seconds=sec)).split(".")
    print(f"End train {times[0]}")
    return best_iou, best_loss
