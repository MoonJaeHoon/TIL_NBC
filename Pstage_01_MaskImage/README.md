# Pstage_image_classification(Mask Image)

- custom_earlystopping.py
  - earlystopping을 적용하기 위한 모듈입니다. 간편하게 직접 customizing할 수 있도록 추가해두었습니다.
- custom_optimizer.py
  - AdamP, SGDP 등 다양한 Optimizer를 사용해보고 싶어 추가한 모듈
- customized_f1score.py
  - LB metric이 f1score이다보니 f1score를 직접 측정하고 저장된 모델을 갱신하는 시기를  f1score가 갱신될 때마다로 하는 것이 효율적이라고 생각하여 추가한 모듈
- dataset.py
  - Data PreProcessing, Data Augmentation, train valid split 등의 기능을 수행하기 위한 모듈
- inference.py
  - 학습한 모델로 submission 파일을 추론하기 위한 모듈
- loss.py
  - focal loss, cross entropy, f1 loss, label smoothing과 같은 loss function이 작성되어있는 모듈
- model.py
  - resnet34, resnet50, resnet101, efficientnet_b4 등의 pretrained backbone 모델을 불러오기 위한 모듈
- train.py
  - Data의 전처리, load부터 모델의 학습을 처음부터 끝까지 수행할 수 있게 모아둔 모듈