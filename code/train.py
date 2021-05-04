import os
import json
import argparse

import torch

from Trainer import Trainer
from dataset import CocoDataset
from build import Model, Criterion, Optimizer, Scheduler, Transform
from utils import fix_random_seed

import neptune.new as neptune
import neptune_config

fix_random_seed(seed=42)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def train(CONFIGS):
    # neptune.
    neptune_run = neptune.init(project="hangjoo/PSTAGE-3", api_token=neptune_config.token, source_files=["code/*.py"],)
    neptune_run["CONFIGS"] = CONFIGS

    # print configs.
    print("\033[31m" + "[ CONFIGS ]" + "\033[0m")
    for c_k, c_v in CONFIGS.items():
        print(f"{c_k} : {c_v}")

    # save path.
    print("\033[31m" + "[ SAVE PATH ]" + "\033[0m")
    save_path = os.path.join("outputs", neptune_run["sys/id"].fetch())
    os.makedirs(save_path, exist_ok=True)
    print(f"All result records would be saved in {save_path}.")

    # save configs in save path.
    with open(os.path.join(save_path, "configs.json"), "w") as config_json:
        json.dump(CONFIGS, config_json, indent=4)

    # transforms.
    transform = Transform(CONFIGS["AUGMENTATION"])

    print("\033[31m" + "[ GENERATE DATASETS ]" + "\033[0m")
    if CONFIGS["K-FOLD_NUM"] > 0:
        # TODO: implement K-Fold dataset code.
        pass
    elif CONFIGS["K-FOLD_NUM"] == 0:
        # data path.
        data_path = "./input/data"
        train_json = os.path.join(data_path, "train.json")
        valid_json = os.path.join(data_path, "valid.json")
        # dataset.
        train_dataset = CocoDataset(
            json_path=train_json,
            data_path=data_path,
            input_size=CONFIGS["INPUT_SIZE"],
            norm_mean=CONFIGS["NORM_MEAN"],
            norm_std=CONFIGS["NORM_STD"],
            mode="train",
            transform=transform,
        )
        valid_dataset = CocoDataset(
            json_path=valid_json,
            data_path=data_path,
            input_size=CONFIGS["INPUT_SIZE"],
            norm_mean=CONFIGS["NORM_MEAN"],
            norm_std=CONFIGS["NORM_STD"],
            mode="valid",
            transform=transform,
        )

    # model.
    model = Model(model_name=CONFIGS["MODEL_NAME"], **CONFIGS["MODEL_PARAMS"]).to(device)

    # criterion.
    criterion = Criterion(criterion_name=CONFIGS["CRITERION_NAME"], **CONFIGS["CRITERION_PARAMS"]).to(device)

    # optimizer.
    if isinstance(CONFIGS["LEARNING_RATE"], dict):
        params = []
        for arc, lr in CONFIGS["LEARNING_RATE"].items():
            arc_cfg = {
                "params": getattr(model, arc).parameters(),
                "lr": lr
            }
            params.append(arc_cfg)
        optimizer = Optimizer(optimizer_name=CONFIGS["OPTIMIZER_NAME"], params=params, **CONFIGS["OPTIMIZER_PARAMS"])
    else:
        optimizer = Optimizer(optimizer_name=CONFIGS["OPTIMIZER_NAME"], params=model.parameters(), lr=CONFIGS["LEARNING_RATE"], **CONFIGS["OPTIMIZER_PARAMS"])

    # scheduler.
    scheduler = Scheduler(scheduler_name=CONFIGS["SCHEDULER_NAME"], optimizer=optimizer, **CONFIGS["SCHEDULER_PARAMS"])

    # training session.
    seg_trainer = Trainer(model=model, criterion=criterion, optimizer=optimizer, scheduler=scheduler, save_path=save_path, neptune_run=neptune_run,)
    seg_trainer.train(
        epoch_num=CONFIGS["EPOCH_NUM"],
        batch_size=CONFIGS["BATCH_SIZE"],
        train_data=train_dataset,
        valid_data=valid_dataset,
        early_stop_num=CONFIGS["EARLY_STOP_NUM"],
        early_stop_target=CONFIGS["EARLY_STOP_TARGET"],
        scheduler_step_type=CONFIGS["SCHEDULER_STEP_TYPE"],
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="code/default_configs.json")
    args = parser.parse_args()

    with open(args.config, "r") as config_f:
        CONFIGS = json.load(config_f)

    train(CONFIGS)
