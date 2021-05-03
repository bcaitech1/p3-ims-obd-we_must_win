import torch
import numpy as np
from utils import get_mIoU, label_accuracy_score


def validation(model, data_loader, criterion, device):
    model.eval()
    with torch.no_grad():
        total_loss = 0
        cnt = 0
        miou_hist = []
        for step, (images, masks) in enumerate(data_loader):
            masks = masks.long()
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            loss = criterion(outputs, masks)
            total_loss += loss
            cnt += 1
            outputs = torch.argmax(outputs, dim=1).detach().cpu().numpy()
            ans_masks = masks.detach().cpu().numpy()
            batch_miou = get_mIoU(outputs, ans_masks, 12)
            miou_hist.extend(batch_miou)
        miou = np.mean(miou_hist)
        avg_loss = total_loss / cnt
    return avg_loss, miou
