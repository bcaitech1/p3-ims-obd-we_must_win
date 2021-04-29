import numpy as np
import torch

from load_data import *

def validation(epoch, model, data_loader, criterion, device):
    print('Start validation #{}'.format(epoch + 1))
    model.eval()
    with torch.no_grad():
        total_loss = 0
        cnt = 0
        mIoU_list = []
        hist = np.zeros((12, 12))
        for step, (images, masks, _) in enumerate(val_loader):
            images = torch.stack(images)
            masks = torch.stack(masks).long()

            images, masks = images.to(device), masks.to(device)

            outputs = model(images)
            loss = criterion(outputs, masks)
            total_loss += loss
            cnt += 1

            outputs = torch.argmax(outputs, dim=1).detach().cpu().numpy()
            hist = add_hist(hist, masks.detach().cpu().numpy(), outputs, n_class=12)
            mIoU = label_accuracy_score(hist)[2]
            mIoU_list.append(mIoU)
        
        avrg_loss = total_loss / cnt
        print('Validation #{}  Average Loss: {:.4f}, mIoU: {:.4f}'.format(epoch, avrg_loss, np.mean(mIoU_list)))
    
    return np.mean(mIoU_list)