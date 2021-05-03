import os
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from utils import get_mIoU, label_accuracy_score

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def collate_fn(batch):
    return tuple(zip(*batch))


class Trainer:
    def __init__(
        self,
        model,
        criterion,
        optimizer,
        scheduler,
        save_path,
        neptune_run=None
    ):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler

        self.save_path = save_path
        self.neptune_run = neptune_run

    def train(
        self,
        epoch_num,
        batch_size,
        train_data,
        valid_data,
        early_stop_num=0,
        early_stop_target="mIoU",
        scheduler_step_type="batch",
    ):
        print("\033[31m" + "[ TRAINING START ]" + "\033[0m")

        # Basic Training.
        if isinstance(train_data, Dataset) and isinstance(valid_data, Dataset):
            train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4, collate_fn=collate_fn)
            valid_loader = DataLoader(valid_data, batch_size=batch_size, shuffle=True, num_workers=4, collate_fn=collate_fn)
            data_loader = {"train": train_loader, "valid": valid_loader}
            self.train_(epoch_num, data_loader, early_stop_num, early_stop_target, scheduler_step_type)

        # K-Fold Training.
        elif isinstance(train_data, list) and isinstance(valid_data, list):
            for fold_idx, (train_data_, valid_data_) in enumerate(zip(train_data, valid_data), 1):
                if not isinstance(train_data_, Dataset) or not isinstance(valid_data_, Dataset):
                    raise Exception("Input train_data, valid_data must be list of torch Dataset Class.")
                # TODO: Implement K-Fold training code.

        else:
            raise Exception("Input train_data, valid_data are of the wrong type. Must be list or single of torch Dataset Class.")

    def train_(self, epoch_num, data_loader, early_stop_num, early_stop_target, scheduler_step_type):
        best_loss = np.inf
        best_mIoU = 0
        early_stop_counter = 0
        for epoch_idx in range(1, epoch_num + 1):
            epoch_time = time.time()

            t_e_result = self.step_(is_train=True, step_idx=epoch_idx, data_loader=data_loader["train"], scheduler_step_type=scheduler_step_type)
            v_e_result = self.step_(is_train=False, step_idx=epoch_idx, data_loader=data_loader["valid"], scheduler_step_type=scheduler_step_type)
            self.save_samples_(step_idx=epoch_idx, data_loader=data_loader["valid"], n_samples=3)

            if self.scheduler is not None and scheduler_step_type == "batch":
                self.scheduler.step()

            time_taken = time.time() - epoch_time
            print(
                f"[Epoch {epoch_idx}] time taken : {time_taken:.2f}" + " " * 30 + "\n"
                f"    train loss : {t_e_result['loss']:.4f} | train mIoU : {t_e_result['mIoU']:.4f}\n"
                f"    valid loss : {v_e_result['loss']:.4f} | valid mIoU : {v_e_result['mIoU']:.4f}",
            )

            best_fits = []
            if v_e_result["loss"] < best_loss:
                torch.save(self.model.state_dict(), os.path.join(self.save_path, "best_loss_model.pth"))
                best_loss = v_e_result["loss"]
                best_fits.append("loss")
            if v_e_result["mIoU"] > best_mIoU:
                torch.save(self.model.state_dict(), os.path.join(self.save_path, "best_mIoU_model.pth"))
                best_mIoU = v_e_result["mIoU"]
                best_fits.append("mIoU")
            if best_fits:
                print(f"    -> Record best validation {'/'.join(best_fits)}. Model weight file saved.")

            if early_stop_target not in best_fits:
                early_stop_counter += 1
            else:
                early_stop_counter = 0

            if self.neptune_run is not None:
                for k, v in t_e_result.items():
                    self.neptune_run[f"Results/train_{k}"].log(value=v, step=epoch_idx)
                for k, v in v_e_result.items():
                    self.neptune_run[f"Results/valid_{k}"].log(value=v, step=epoch_idx)

            if early_stop_counter == early_stop_num:
                print("    -> Early Stopped.")
                return

    def step_(self, is_train, step_idx, data_loader, scheduler_step_type):
        loss_hist = []
        miou_hist = []

        if is_train:
            self.model.train()
        else:
            self.model.eval()

        with torch.set_grad_enabled(is_train):
            for iter_idx, (images, masks, _) in enumerate(data_loader, 1):
                images = torch.stack(images).to(device)
                masks = torch.stack(masks).to(device)

                outputs = self.model(images)
                loss = self.criterion(outputs, masks)

                if is_train:
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    if self.scheduler is not None and scheduler_step_type == "batch":
                        self.scheduler.step()

                pred_masks = torch.argmax(outputs, dim=1).detach().cpu().numpy()
                ans_masks = masks.detach().cpu().numpy()
                batch_mIoU = get_mIoU(pred_masks, ans_masks, n_class=12)

                loss_hist.append(loss.item())
                miou_hist.extend(batch_mIoU)
                print(f"[Epoch {step_idx}] {'train' if is_train else 'valid'} iteration - {iter_idx}/{len(data_loader)}" + " " * 10, end="\r")

        loss = np.mean(loss_hist)
        mIoU = np.mean(miou_hist)

        result = {"loss": loss, "mIoU": mIoU}

        return result

    def save_samples_(self, step_idx, data_loader, n_samples=3):
        os.makedirs(os.path.join(self.save_path, f"epoch_{step_idx}_samples"))

        sample_idx = np.random.choice(len(data_loader.dataset), n_samples)
        sample_images, ground_truths, _ = data_loader.dataset[sample_idx]

        norm_mean = data_loader.dataset.normalizer.transforms.transforms[0].mean
        norm_std = data_loader.dataset.normalizer.transforms.transforms[0].std

        with torch.no_grad():
            predicts = self.model(torch.stack(sample_images).to(device))
            predicts = torch.argmax(predicts, dim=1)
            predicts = predicts.cpu().numpy()  # predicted mask images. (n_samples, height, width)

            for iter_idx, (image, predict, gt) in enumerate(zip(sample_images, predicts, ground_truths)):
                # restore normalized image.
                image = image.numpy().transpose(1, 2, 0)
                image = (image * norm_std) + norm_mean
                image = (image * 255).astype(np.uint8)

                gt = gt.numpy()

                plt.figure(figsize=(18, 5))

                plt.subplot(1, 3, 1)
                plt.axis("off")
                plt.title("Original Image")
                plt.imshow(image)

                plt.subplot(1, 3, 2)
                plt.axis("off")
                plt.title("Predicted Mask")
                plt.imshow(predict)

                plt.subplot(1, 3, 3)
                plt.axis("off")
                plt.title("Ground Truth")
                plt.imshow(gt)

                fig_path = os.path.join(self.save_path, f"epoch_{step_idx}_samples", f"sample_image_{iter_idx}.jpg")
                plt.savefig(fig_path)

                print(f"[Epoch {step_idx}] saving sample images iteration - {iter_idx}/{len(data_loader)}" + " " * 10, end="\r")
