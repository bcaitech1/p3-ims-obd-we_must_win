## P-Stage 3
### Object Detection

- Model: DetectoRS
- Pretrain: ImageNet
- Optimization: AdamW
- Loss: CrossEntropy
- Learning late: 1.00E-04
- Schedule:	StepLR(optimizer, 15, gamma=0.1)
- Augmentation: Normalize, HorizontalFlip, Instaboost
- Result: 0.5318
- TTA: multi test scale, flip