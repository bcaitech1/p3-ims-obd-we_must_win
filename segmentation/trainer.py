import os
import torch

import numpy as np

from utils import label_accuracy_score, label_accuracy_score_original
from utils import get_mIoU

def train(num_epochs, model, model_name, train_loader, val_loader, criterion, optimizer, patience, val_every, device):

    print('Start training..')
    best_loss = 9999999
    best_mIoU = 0
    counter = 0
    for epoch in range(num_epochs):
        model.train()
        for step, (images, masks, _) in enumerate(train_loader):
            images = torch.stack(images)       # (batch, channel, height, width)
            masks = torch.stack(masks).long()  # (batch, channel, height, width)
            
            # gpu 연산을 위해 device 할당
            images, masks = images.to(device), masks.to(device)
                  
            # inference
            outputs = model(images)
            
            # loss 계산 (cross entropy loss)
            loss = criterion(outputs, masks)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # step 주기에 따른 loss 출력
            if (step + 1) % 25 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(
                    epoch+1, num_epochs, step+1, len(train_loader), loss.item()))
        
        # validation 주기에 따른 loss 출력 및 best model 저장
        if (epoch + 1) % val_every == 0:
            avrg_loss, mIoU = validation(epoch + 1, model, val_loader, criterion, device)
            
            if avrg_loss < best_loss:
                best_loss = avrg_loss

            if mIoU > best_mIoU:
                best_mIoU = mIoU        
                print('Best performance at epoch: {}'.format(epoch + 1))
                print('best mIoU : {:.4f}'.format(best_mIoU))
                print('Saved model')

                save_model(model, file_name= model_name + "_" + str(best_mIoU)[:6] + ".pt")
                counter = 0
            else:
                counter += 1

            # Callback2: patience 횟수 동안 성능 향상이 없을 경우 학습을 종료시킵니다.
            if counter > patience:
                print("Early Stopping...")
                return best_mIoU



# ======= 검증 ======= #
def validation(epoch, model, data_loader, criterion, device):
    print('Start validation #{}'.format(epoch))
    model.eval()
    with torch.no_grad():
        total_loss = 0
        cnt = 0
        mIoU_hist = []
        loss_hist = []
        for step, (images, masks, _) in enumerate(data_loader):
            
            images = torch.stack(images)       # (batch, channel, height, width)
            masks = torch.stack(masks).long()  # (batch, channel, height, width)

            images, masks = images.to(device), masks.to(device)            

            outputs = model(images)
            loss = criterion(outputs, masks)
            total_loss += loss
            cnt += 1
            
            outputs = torch.argmax(outputs.squeeze(), dim=1).detach().cpu().numpy()
            masks_detach = masks.detach().cpu().numpy()

            batch_mIoU = get_mIoU(outputs, masks_detach, 12)

            loss_hist.append(loss.item())
            mIoU_hist.extend(batch_mIoU)

        mIoU = np.mean(mIoU_hist)
        avrg_loss = np.mean(loss_hist)


        print('Validation #{}  Average Loss: {:.4f}, mIoU: {:.4f}'.format(epoch, avrg_loss, np.mean(mIoU_hist)))

    return avrg_loss, mIoU




def save_model(model, file_name='fcn8s_best_model(pretrained).pt'):
    saved_dir = '../saved'
    if not os.path.isdir(saved_dir):                                                           
        os.mkdir(saved_dir)

    check_point = {'net': model.state_dict()}
    output_path = os.path.join(saved_dir, file_name)
    torch.save(model.state_dict(), output_path)