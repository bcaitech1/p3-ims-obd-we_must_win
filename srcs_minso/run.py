import argparse
from train import trainer
from utils import set_seed
from utils import set_device
from model import get_model
from eval import inference
import os
import pandas as pd


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_lib", type=bool, default=True, help="use segmentation_models_pytorch(default:True)")
    parser.add_argument("--architecture", type=str, default="DeepLabV3Plus", help="segmentation architecture")
    parser.add_argument("--backbone", type=str, default="resnext50_32x4d", help="backbone model")
    parser.add_argument("--backbone_weight", type=str, default="swsl", help="pretrain weight")
    parser.add_argument("--epochs", type=int, default=30, help="number of epochs(default:50)")
    parser.add_argument("--early_stop", type=bool, default=True, help="use early stop(default:True")
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate(default:1e-4)")
    parser.add_argument("--seed", type=int, default=42, help="input random seed(default:42)")
    parser.add_argument("--batch_size", type=int, default=16, help="input batch size(default:16)")
    parser.add_argument("--kfold", type=int, default=0, help="0:not use, 1:Kfold, 2:StratifiedKfold")
    parser.add_argument("--transform", type=int, default=0)
    parser.add_argument("--save_dir", type=str, default="/opt/ml/code/saved")
    parser.add_argument("--width", type=int, default=256, help="resize width")
    parser.add_argument("--height", type=int, default=256, help="resize height")
    parser.add_argument("--id", type=int, default=0)
    parser.add_argument("--scheduler", type=int, default=0)
    parser.add_argument("--optimizer", type=int, default=0)
    args = parser.parse_args()
    return args


def main():
    args = parse()
    device = set_device()
    set_seed(args.seed)
    best_iou = trainer(args, device)
    inference(args, device)
    fields = ["architecture", "backbone", "lr", "batch_size", "aug_option",  "best_iou"]
    if not os.path.exists("log.csv"):
        data = [(args.architecture, args.backbone, args.lr, args.batch_size, args.transform, best_iou)]
        df = pd.DataFrame(data, columns=fields)
    else:
        data = [args.architecture, args.backbone, args.lr, args.batch_size, args.transform, best_iou]
        df = pd.read_csv("log.csv")
        df = df.append(pd.Series(data, index=df.columns), ignore_index=True)
    df.to_csv("log.csv", index=False)


if __name__ == "__main__":
    main()
