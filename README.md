# Hangjoo Codes.

## How to train model.

**주의 :** train.py 내에 neptune API를 사용하는 코드가 있습니다. neptune API는 API KEY를 필요로 하기 때문에 오류를 발생시킵니다. neptune API를 사용하지 않는 경우 해당 라인을 주석 처리하고 사용하셔야 합니다.

1. `code/default_configs.json` 파라미터를 수정합니다.

    사용가능한 `{MODEL/CRITERION/SCHEDULER}_NAME`은 `code/build.py`에서 확인하실 수 있습니다.
    현재 사용가능한 Model, Criterion, Scheduler 목록은 아래와 같습니다.

    ```python
    model_list = {
        "Unet": smp.Unet,
        "FPN": smp.FPN,
        "DeepLabV3": smp.DeepLabV3,
        "DeepLabV3+": smp.DeepLabV3Plus,
        "PAN": smp.PAN,
        "HRNet": HRNet,
    }

    criterion_list = {
    "CrossEntropyLoss": nn.CrossEntropyLoss,
    "MSE": nn.MSELoss,
    "FocalLoss": FocalLoss,
    "KLDiv": nn.KLDivLoss,
    "LabelSmoothingLoss": LabelSmoothingLoss,
    "DiceLossWithCEE": DiceLossWithCEE
    }

    optimizer_list = {
        "Adam": Adam,
        "SGD": SGD,
        "MADGRAD": MADGRAD,
        "AdamP": AdamP,
        "SGDP": SGDP,
        "AdamW": AdamW,
        "RAdam": RAdam,
    }

    scheduler_list = {
        "CosineAnnealingLR": CosineAnnealingLR,
        "StepLR": StepLR,
        "MultiStepLR": MultiStepLR,
    }

    transform_list = {
        "HorizontalFlip": A.HorizontalFlip,
        "VerticalFlip": A.VerticalFlip,
        "GridMask": GridMask,
        "RandomBrightnessContrast": A.RandomBrightnessContrast,
        "Rotate": A.Rotate,
    }
    ```

    `{MODEL/CRITERION/SCHEDULER}_PARAMS` : 학습 모델, 손실 함수, 스케쥴러를 생성할 때 추가로 전달할 인자를 정의할 수 있습니다.
    예를 들어 아래와 같이 설정된 CONFIG은 아래와 같은 모델을 생성합니다.

    ```json
    "MODEL_NAME": "DeepLabV3+",
    "MODEL_PARAMS": {
        "encoder_name": "resnext101_32x16d",
        "encoder_weights": "swsl",
        "classes": 12
    }
    ```

    ```python
    model = Model("DeepLabV3+", encoder_name="resnext101_32x16d", encoder_weights="swsl", classes=12)
    ```

    `AUGMENTATION` : Augmentation을 정의하는 문자열 혹은 문자열 리스트가 가능합니다. 아래와 같은 AUGMENTATION 인자는 아래 코드와 같이 생성됩니다.

    ```json
    "AUGMENTATION": [
        "A.NoOp()"
        ["A.Rotate(limit=45)", "A.GridDropout(ratio=0.25)", "A.RandomBrightnessContrast()"],
        "A.Resize(height=256, width=256)"
    ]
    ```

    ```python
    transform = A.OneOf([
        A.NoOp(),
        A.Compose([A.Rotate(limit=45), A.GridDropout(ratio=0.25), A.RandomBrightnessContrast()]),
        A.Resize(height=256, width=256),
    ])
    ```

    

    `K-FOLD_NUM` : 0의 경우 설정된 학습 데이터와 검증 데이터로 학습을 진행합니다. 1 이상의 정수 값을 갖는 경우 K-FOLD_NUM만큼의 FOLD를 생성하여 학습과 검증 과정을 반복합니다.

    `EARLY_STOP_NUM` : 학습 시 Early Stop에 대한 Threshold 값입니다. 해당 인자 값만큼 Target Value가 갱신되지 않을 경우 학습이 종료됩니다.

    `EARLY_STOP_TARGET` : Early Stop을 위한 갱신 Target Value 입니다. "loss" 혹은 "mIoU"로 설정할 수 있습니다.

    `GRADIENT_ACCUMULATE_STEP' : 설정시 Gradient를 누적하고 있다가 GRADIENT_ACCUMULATE_STEP의 step 마다 parameter optimization이 적용됩니다.

    

2. 
    train.py를 실행합니다. 해당`default_configs.json`을 유지한 상태로 새로 추가한 CONFIG을 사용하고 싶은 경우 아래와 같이 train.py를 실행시킬 수 있습니다. 이 때 `default_configs.json`에 있는 인자는 모두 포함한 파일이여야 합니다.

    ```bash
    python train.py --configs {CONFIG_PATH}
    ```

3. 학습 결과 파일은 `outputs` 폴더에 생성됩니다. 해당 폴더 내에 `PST-{id_num}`로 생성 되며 해당 학습에 설정된 configs.json 파일, 매 epoch마다 생성된 샘플 이미지와 학습하면서 가장 좋은 validation loss와 mIoU 점수를 기록한 모델 파일이 저장됩니다.

## How to inference.

