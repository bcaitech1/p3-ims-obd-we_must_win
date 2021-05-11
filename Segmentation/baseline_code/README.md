# P Stage3 Segmentation

## Usage

### Training

```python
python train.py [optional arguments]
```

### optional arguments

`--seed` (type=int, default=0)

`--batch_size` (type=int, default=32)

`--epochs` (type=int, default=50)

`--lr` (type=float, default=1e-4)

`--model_type` (type=str, default='Unet')

`--backbone` (type=str, default='efficientnet-b0')

`--pretrained_weight` (type=str, default='imagenet')

`--loss_fn` (type=str, default='CrossEntropyLoss')

`--optim` (type=str, default='AdamW')

`--weight_decay` (type=float, default=1e-6)

`--saved_dir` (type=str, default='./saved')

**For example, to train `DeepLabV3` model with a `resnet50` backbone**

```python
python train.py --model_type DeepLabV3 --backbone resnet50
```

### Inference

```python
python inference.py [optional arguments]
```

### optional arguments

`--batch_size` (type=int, default=32)

`--out_path` (type=str, default='./submission/submission.csv')

`--model_type` (type=str, default='Unet')

`--backbone` (type=str, default='efficientnet-b0')

`--pretrained_weight` (type=str, default='imagenet')

---

### [Reference](https://github.com/qubvel/segmentation_models.pytorch#models)