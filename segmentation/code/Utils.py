# 첫번째 batch의 추론 결과 확인
import torch
import matplotlib.pyplot as plt
# https://github.com/wkentaro/pytorch-fcn/blob/master/torchfcn/utils.py
import numpy as np
import os


import math
import torch
from torch.optim.lr_scheduler import _LRScheduler


device = "cuda" if torch.cuda.is_available() else "cpu"
def inference_view(model, dataset, idx, result_plot=True,confidence_plot=True, ground_truth=False,
                   model_path='model_name', category = 'battery'): # inference_view
    COLORS = [
        [129, 236, 236],
        [2, 132, 227],
        [232, 67, 147],
        [255, 234, 267],
        [0, 184, 148],
        [85, 239, 196],
        [48, 51, 107],
        [255, 159, 26],
        [255, 204, 204],
        [179, 57, 57],
        [248, 243, 212],
    ]
    COLORS = np.vstack([[0, 0, 0], COLORS]).astype('uint8')

    model.to(device)
    category_names = dataset.category_names
    if ground_truth:
        image, gt, image_infos = dataset[idx]
    else:
        image, image_infos = dataset[idx]
    model.eval()
    # inference
    outs = model(image.unsqueeze(0).to(device))
    oms = torch.argmax(outs, dim=1).detach().cpu().numpy()


    save_dir = model_path + '/' + category + '/'
    try:
        os.mkdir(model_path)
    except:
        pass

    try:
        os.mkdir(save_dir)
    except:
        pass
    if result_plot:
        if ground_truth:
            fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(16, 16))

            print('Shape of Original Image :', list(image.shape))
            print('Shape of Predicted : ', list(oms.shape))
            print('Unique values, category of transformed mask : \n',
                  [{i, category_names[int(i)]} for i in list(np.unique(oms))])
            print('ground Truth',
                  [{i, category_names[int(i)]} for i in list(np.unique(gt))])

            # Original image
            ax1.imshow(image.permute([1, 2, 0]))
            ax1.grid(False)
            ax1.set_title("Original image : {}".format(image_infos['file_name']), fontsize=15)

            # ground_Truth
            filtered_gt = COLORS[gt.detach().cpu().numpy().astype('int8')]
            ax2.imshow(filtered_gt)
            ax2.grid(False)
            ax2.set_title("ground_Truth : {}".format(image_infos['file_name']), fontsize=15)

            # Predicted
            filtered_oms = COLORS[oms[0].astype('int8')]
            ax3.imshow(filtered_oms)
            ax3.grid(False)
            ax3.set_title("Predicted : {}".format(image_infos['file_name']), fontsize=15)


        else:
            fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(16, 16))

            print('Shape of Original Image :', list(image.shape))
            print('Shape of Predicted : ', list(oms.shape))
            print('Unique values, category of transformed mask : \n',
                  [{i, category_names[int(i)]} for i in list(np.unique(oms))])

            # Original image
            ax1.imshow(image.permute([1, 2, 0]))
            ax1.grid(False)
            ax1.set_title("Original image : {}".format(image_infos['file_name']), fontsize=15)

            # Predicted
            filtered_oms = COLORS[gt.detach().cpu().numpy().astype('int8')]
            ax2.imshow(filtered_oms)
            ax2.grid(False)
            ax2.set_title("Predicted : {}".format(image_infos['file_name']), fontsize=15)
    if ground_truth:
        plt.savefig(f'{save_dir}train_{idx}_predict')
    else:
        plt.savefig(f"{save_dir}test_{idx}_predict")

    if confidence_plot:
        fig, axes = plt.subplots(nrows=3, ncols=4, figsize=(20, 10))
        axes = axes.flatten()
        for i, c in enumerate(outs[0]):
            axes[i].imshow(c.detach().cpu().numpy())
            axes[i].set_title(f"{i, category_names[int(i)]}")

    if ground_truth:
        plt.savefig(f'{save_dir}train_{idx}_logits')
    else:
        plt.savefig(f"{save_dir}test_{idx}_logits")
    plt.show()
    return outs, oms




class CosineAnnealingWarmupRestarts(_LRScheduler):
    """
        optimizer (Optimizer): Wrapped optimizer.
        first_cycle_steps (int): First cycle step size.
        cycle_mult(float): Cycle steps magnification. Default: -1.
        max_lr(float): First cycle's max learning rate. Default: 0.1.
        min_lr(float): Min learning rate. Default: 0.001.
        warmup_steps(int): Linear warmup step size. Default: 0.
        gamma(float): Decrease rate of max learning rate by cycle. Default: 1.
        last_epoch (int): The index of last epoch. Default: -1.
    """

    def __init__(self,
                 optimizer: torch.optim.Optimizer,
                 first_cycle_steps: int,
                 cycle_mult: float = 1.,
                 max_lr: float = 0.1,
                 min_lr: float = 0.001,
                 warmup_steps: int = 0,
                 gamma: float = 1.,
                 last_epoch: int = -1
                 ):
        assert warmup_steps < first_cycle_steps

        self.first_cycle_steps = first_cycle_steps  # first cycle step size
        self.cycle_mult = cycle_mult  # cycle steps magnification
        self.base_max_lr = max_lr  # first max learning rate
        self.max_lr = max_lr  # max learning rate in the current cycle
        self.min_lr = min_lr  # min learning rate
        self.warmup_steps = warmup_steps  # warmup step size
        self.gamma = gamma  # decrease rate of max learning rate by cycle

        self.cur_cycle_steps = first_cycle_steps  # first cycle step size
        self.cycle = 0  # cycle count
        self.step_in_cycle = last_epoch  # step size of the current cycle

        super(CosineAnnealingWarmupRestarts, self).__init__(optimizer, last_epoch)

        # set learning rate min_lr
        self.init_lr()

    def init_lr(self):
        self.base_lrs = []
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.min_lr
            self.base_lrs.append(self.min_lr)

    def get_lr(self):
        if self.step_in_cycle == -1:
            return self.base_lrs
        elif self.step_in_cycle < self.warmup_steps:
            return [(self.max_lr - base_lr) * self.step_in_cycle / self.warmup_steps + base_lr for base_lr in
                    self.base_lrs]
        else:
            return [base_lr + (self.max_lr - base_lr) \
                    * (1 + math.cos(math.pi * (self.step_in_cycle - self.warmup_steps) \
                                    / (self.cur_cycle_steps - self.warmup_steps))) / 2
                    for base_lr in self.base_lrs]

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
            self.step_in_cycle = self.step_in_cycle + 1
            if self.step_in_cycle >= self.cur_cycle_steps:
                self.cycle += 1
                self.step_in_cycle = self.step_in_cycle - self.cur_cycle_steps
                self.cur_cycle_steps = int(
                    (self.cur_cycle_steps - self.warmup_steps) * self.cycle_mult) + self.warmup_steps
        else:
            if epoch >= self.first_cycle_steps:
                if self.cycle_mult == 1.:
                    self.step_in_cycle = epoch % self.first_cycle_steps
                    self.cycle = epoch // self.first_cycle_steps
                else:
                    n = int(math.log((epoch / self.first_cycle_steps * (self.cycle_mult - 1) + 1), self.cycle_mult))
                    self.cycle = n
                    self.step_in_cycle = epoch - int(
                        self.first_cycle_steps * (self.cycle_mult ** n - 1) / (self.cycle_mult - 1))
                    self.cur_cycle_steps = self.first_cycle_steps * self.cycle_mult ** (n)
            else:
                self.cur_cycle_steps = self.first_cycle_steps
                self.step_in_cycle = epoch

        self.max_lr = self.base_max_lr * (self.gamma ** self.cycle)
        self.last_epoch = math.floor(epoch)
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr




def _fast_hist(label_true, label_pred, n_class):
    mask = (label_true >= 0) & (label_true < n_class)
    hist = np.bincount(n_class * label_true[mask].astype(int) + label_pred[mask],
                        minlength=n_class ** 2).reshape(n_class, n_class)
    return hist


def label_accuracy_score(hist):
    """
    Returns accuracy score evaluation result.
      - [acc]: overall accuracy
      - [acc_cls]: mean accuracy
      - [mean_iu]: mean IU
      - [fwavacc]: fwavacc
    """
    acc = np.diag(hist).sum() / hist.sum()
    with np.errstate(divide='ignore', invalid='ignore'):
        acc_cls = np.diag(hist) / hist.sum(axis=1)
    acc_cls = np.nanmean(acc_cls)

    with np.errstate(divide='ignore', invalid='ignore'):
        iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
    mean_iu = np.nanmean(iu)

    freq = hist.sum(axis=1) / hist.sum()
    fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
    return acc, acc_cls, mean_iu, fwavacc


def add_hist(hist, label_trues, label_preds, n_class):
    """
        stack hist(confusion matrix)
    """

    for lt, lp in zip(label_trues, label_preds):
        hist += _fast_hist(lt.flatten(), lp.flatten(), n_class)

    return hist