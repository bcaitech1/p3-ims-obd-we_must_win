# Segmentation

모델 및 pretrain weight는 https://github.com/qubvel/segmentation_models.pytorch를 사용하였습니다.

# Code 

### run.py

- Parameter
  - use_lib : segmentation_models(smp) 라이브러리 사용 유무, False의 경우 직접 구현한 모델 사용 -> 아직 구현 X
  - architecture : smp에서 사용할 architecture
  - backbone : architecture에 사용할 backbone
  - kfold : kfold사용 유무 -> 0일경우 kfold사용 x, 1일경우 stratifiedkfold
  - scheduler : 0 -> StepLR, 1 -> CosineAnnealingLR, 2 -> CosineAnnealingWarnRestrats

### train.py

- Kfold 사용유무, scheduler option, model에 따라 학습 진행

### dataset.py

- Kfold 사용하지 않을 때 사용할 MyDataset class, Kfold를 위한 CsvDataset Class
- MyDataset Class
  - pycocotools를 사용하여 transform을 적용한 image, output을 반환하는 Dataset
- CsvDataset
  - StratifiedKfold사용을 위해 이미지별 대표 label을 사용
  - 대표 label의 경우 전체 data기준 각 label의 분포를 계산한 뒤 각 이미지에 포함된 label중 전체 data기준 분포가 가장 작은 label을 대표 label로 설정 

### model.py

- get_model
  - smp 라이브러리를 사용해 Model을 반환하는 함수

### validation.py

- mIoU 계산을 위한 함수

### eval.py

- 최종 submission 파일 생성을 위한 함수들

### utils.py

- 학습 및 검증에 사용할 utils함수들 