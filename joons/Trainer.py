import os
from functools import partial
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR, StepLR

import numpy as np
import random

from Datasets import RecylceDatasets
from Utils import CosineAnnealingWarmupRestarts,label_accuracy_score,add_hist

from Models import *
from SwinTransformer import *

import albumentations as A
from albumentations.pytorch import ToTensorV2
import wandb


def save_model(model, file_name='deeplabv3_resnet101(pretrained)'):
    check_point = {'net': model.state_dict()}
    file_name = file_name + '.pt'
    output_path = os.path.join('/opt/ml/code/saved', file_name)
    torch.save(model.state_dict(), output_path)

class Trainer():
    def __init__(self,
                 num_epochs, model,
                 criterion, optimizer,
                 batch_size, learning_rate, weight_decay,
                 saved_dir, save_step,
                 random_seed = 42,
                 train_dir = "/opt/ml/input/data/train.json",
                 val_dir = "/opt/ml/input/data/val.json",
                 train_transform = None,
                 test_transform = None,
                 resize = None,
                 scheduler = None,
                 ):

        self.random_seed = random_seed
        self.set_seed(random_seed)
        self.num_epochs = num_epochs
        self.scheduler = scheduler

        if model == 'deeplabv3_resnet101':
            self.model = Deeplabv3ResNet101()
        elif model == 'deeplabv3_resnet34':
            self.model = Deeplabv3ResNet34()
        elif model == 'deeplabv3_resnext101_32x16d':
            self.model = Deeplabv3ResNext101_32x16d()
        elif model == 'deeplabv3_resnext101_32x4d':
            self.model = Deeplabv3resnext101_32x4d()
        elif model == 'deeplabv3_resnext50_32x4d':
            self.model = Deeplabv3resnext50_32x4d()
        elif model == 'deeplabv3+_resnext50_32x4d':
            self.model = Deeplabv3Plus_resnext50_32x4d()
        elif model == 'deeplabv3+_effinetb0':
            self.model = Deeplabv3Plus_efficientnetb0()
        elif model == 'swin_transformer_base':
            self.model = SwinTransformerBase()
        elif model == 'swin_transformer_small':
            self.model = SwinTransformerSmall()
        elif model == 'swin_transformer_tiny':
            self.model = SwinTransformerTiny()
        else:
            self.model = model

        self.train_transform = train_transform
        self.test_transform = test_transform
        self.batch_size = batch_size

        self.size = resize
        self.train_loader, self.val_loader = self.get_dataloader(train_dir=train_dir, val_dir=val_dir)
        self.step_size = len(self.train_loader)

        self.criterion = criterion

        self.optimizer = optimizer(params = self.model.parameters(),
                                   lr = learning_rate,
                                   weight_decay=weight_decay)
        if self.scheduler:
            self.scheduler = self.scheduler(optimizer=self.optimizer)

        self.save_dir = saved_dir
        self.save_step = save_step

        self.device = "cuda" if torch.cuda.is_available() else "cpu"


    def train(self):
        print('Start training..')
        self.model.to(self.device)
        best_loss = 9999999
        best_miou = 0.0
        for epoch in range(self.num_epochs):
            self.model.train()
            for step, (images, masks, _) in enumerate(self.train_loader):
                images = torch.stack(images)  # (batch, channel, height, width)
                masks = torch.stack(masks).long()  # (batch, channel, height, width)

                # gpu 연산을 위해 device 할당
                images, masks = images.to(self.device), masks.to(self.device)

                # inference
                outputs = self.model(images)

                # loss 계산 (cross entropy loss)
                loss = criterion(outputs, masks)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                if self.scheduler:
                    self.scheduler.step()

                # step 주기에 따른 loss 출력
                if (step + 1) % 25 == 0:
                    print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(
                        epoch + 1, self.num_epochs, step + 1, self.step_size, loss.item()))
                    if self.scheduler:
                        wandb.log({'epoch':epoch+1, 'loss':loss.item(), 'learning_rate':self.scheduler.get_last_lr()[0]},
                                  step=step+(len(self.train_loader)*epoch))
                    else:
                        wandb.log({'epoch': epoch + 1, 'loss': loss.item()},
                                  step=step + (len(self.train_loader) * epoch))

            # validation 주기에 따른 loss 출력 및 best model 저장
            if (epoch + 1) % self.save_step == 0:
                avrg_loss, mIoU = self.validation(epoch+1)
                if avrg_loss < best_loss:
                    print('Best performance(Loss) at epoch: {}'.format(epoch + 1))
                    print('Save model in', self.save_dir)
                    best_loss = avrg_loss
                    save_model(self.model, self.save_dir)

                    with open(f'./saved/{self.save_dir}_model_best_Loss_epoch.txt', 'w') as f:
                        f.write(f"{epoch + 1}")

                if mIoU > best_miou:
                    print('Best performance(mIoU) at epoch: {}'.format(epoch + 1))
                    print('Save model in', self.save_dir)
                    best_miou = mIoU
                    save_model(self.model, 'mIoU_'+self.save_dir)

                    with open(f'./saved/{self.save_dir}_model_best_mIou_epoch.txt', 'w') as f:
                        f.write(f"{epoch + 1}")


    def validation(self, epoch):
        n_class = 12
        print('Start validation #{}'.format(epoch))
        self.model.eval()
        mIoU_list = []
        with torch.no_grad():
            total_loss = 0
            cnt = 0
            mIoU_list = []
            for step, (images, masks, _) in enumerate(self.val_loader):
                images = torch.stack(images)  # (batch, channel, height, width)
                masks = torch.stack(masks).long()  # (batch, channel, height, width)

                images, masks = images.to(self.device), masks.to(self.device)

                outputs = self.model(images)
                loss = self.criterion(outputs, masks)
                total_loss += loss
                cnt += 1


                outputs = torch.argmax(outputs.squeeze(), dim=1).detach().cpu().numpy()

                hist = np.zeros((n_class, n_class))
                hist = add_hist(hist, masks.detach().cpu().numpy(), outputs, n_class=n_class)
                # mIoU = label_accuracy_score(masks.detach().cpu().numpy(), outputs, n_class=12)[2]
                # mIoU_list.append(mIoU)
                acc, acc_cls, mIoU, fwavacc = label_accuracy_score(hist)
                mIoU_list.append(mIoU)
            avrg_loss = total_loss / cnt
            mIoU = np.mean(mIoU_list)
            print('Validation #{}  Average Loss: {:.4f}, mIoU: {:.4f}'.format(epoch, avrg_loss, mIoU))
            wandb.log({'test_accuracy': mIoU, 'test_loss' : avrg_loss },
                      step= step+ (len(self.train_loader)*epoch))
        return avrg_loss, mIoU


    def get_dataloader(self, train_dir, val_dir): # data_loader
        train_dataset = RecylceDatasets(train_dir, mode='train', transform=self.train_transform, resize=self.size)
        val_dataset = RecylceDatasets(val_dir, mode='val', transform=self.test_transform, resize=self.size)

        def collate_fn(batch):
            return tuple(zip(*batch))


        train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                   batch_size=self.batch_size,
                                                   shuffle=True,
                                                   num_workers=4,
                                                   collate_fn=collate_fn)

        val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                                 batch_size=self.batch_size,
                                                 shuffle=False,
                                                 num_workers=4,
                                                 collate_fn=collate_fn)
        return train_loader, val_loader

    def set_seed(self, random_seed):
        # seed 고정
        torch.manual_seed(random_seed)
        torch.cuda.manual_seed(random_seed)
        # torch.cuda.manual_seed_all(random_seed) # if use multi-GPU
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(random_seed)
        random.seed(random_seed)






if __name__ == "__main__":
    import torch
    import torch.nn as nn
    saved_dir = './saved'
    if not os.path.isdir(saved_dir):
        os.mkdir(saved_dir)

    # custom_model = Deeplabv3resnext50_32x4d()
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    # model_path = '/opt/ml/code/saved/deeplabv3_resnext50_32x4d_cosine_annealing_constant.pt'
    # checkpoint = torch.load(model_path, map_location=device)
    # custom_model.load_state_dict(checkpoint)



    ## cosine_annealing
    cosine_annealing_scheduler_arg = dict(
        first_cycle_steps=400,
        cycle_mult=1.0,
        max_lr=2e-04,
        min_lr=1e-07,
        warmup_steps=30,
        gamma=0.8
    )
    # scheduler = partial(CosineAnnealingWarmupRestarts, **cosine_annealing_scheduler_arg)


    ## stepLR
    stepLR_arg = dict(
        step_size = 378*2,
        gamma = 0.7

    )
    scheduler = partial(StepLR, **stepLR_arg)


    ## linear schedule with Warmup
    warmup_linear = dict(
        max_lr=3e-04, steps_per_epoch=164, epochs=50,
        anneal_strategy='linear', pct_start=0.15
    )
    # scheduler = partial(OneCycleLR, **warmup_linear)



    ## transform
    train_transform = A.Compose([
        # A.CLAHE(p=1),
        # A.RandomGridShuffle(),
        A.Rotate(),
        A.HorizontalFlip(),
        ToTensorV2()
    ])

    test_transform = A.Compose([
        ToTensorV2()
    ])
    # Loss function 정의
    criterion = nn.CrossEntropyLoss()


    # Optimizer 정의
    ## Adam
    # opt = torch.optim.Adam
    ## SGD
    # opt = partial( torch.optim.SGD, momentum = 0.5 )
    ## AdamP
    # opt = partial( AdamP, betas=(0.9, 0.999))
    ## AdamW
    opt = AdamW

    config = dict(
        num_epochs=50,
        model='swin_transformer_base',
        criterion=criterion,
        optimizer=opt,
        batch_size=7,
        learning_rate=2e-04,
        weight_decay=1e-02,
        # saved_dir="swin_base_512_stepRL",
        saved_dir = 'swin_transformer_base_512',
        save_step=1,
        train_dir="/opt/ml/input/data/train.json",
        val_dir="/opt/ml/input/data/val.json",
        train_transform=train_transform,
        test_transform = test_transform,
        # resize = 512,
        random_seed = 42,
        scheduler = scheduler

    )

    with wandb.init(project="Semantic_Segmentation", config=config):
        # access all HPs through wandb.config, so logging matches execution!
        # config = wandb.config
        wandb.run.name = config['saved_dir']

        trainer=Trainer(
           **config
        )
        trainer.train()



