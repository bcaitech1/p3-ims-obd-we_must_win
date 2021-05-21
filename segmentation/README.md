## P-Stage 3
### Segementation

- Model: DeepLabV3+ + resnext101_32x16d
- Pretrain: swsl
- Optimization: AdamW
- Loss: CrossEntropy
- Learning late: 1.00E-04
- Schedule:	StepLR(optimizer, 15, gamma=0.1)
- Augmentation: Normalize, HorizontalFlip, Rotate, RandomGridShuffle, OpticalDistortion, Cutout
- Result: 0.6714
- multi test scale, pseudo labeling