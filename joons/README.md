### Stage3-1 Semantic Segmentation

##### We Must Win!
<br></br>

#### python files
- Datasets.py : 데이터셋 관련 코드
- Models.py : 실험한 모델들을 모아놓은 코드
- Utils.py : CosineAnnealing(decrease), inference결과를 보여주는 기능을 구현한 코드
- Trainer.py : 모델 학습 구현 코드 (다양한 실험을 할시 위 파일의 main읉 수정하면 됩니다.)
- Inference.py : 학습된 모델을 바탕으로 추론 및 결과 제출

<br></br>

#### ipynb files
- EDA.ipynb : Dataset EDA
- Ensemble.ipynb : 앙상블 코드, TTA기능 구현

<br></br>

#### SwinTransformers
Swin_Transformer 모델 구현한 모듈( mmseg 라이브러리 활용 )

<br></br>

### Demo
```
python Trainer.py
```


