# 1. 과제2 코드에 나온 torch.nn.Upsample에 대하여 궁금한 점이 있습니다.

안녕하세요. 과제2를 수행하고 나서 과제2 코드에 나온 torch.nn.Upsample에 대하여 궁금한 것이 생겨서 질문을 올리게 되었습니다.



제가 이해한 내용으로는 torch.nn.Upsample는 미리 정의된 보간법으로 학습 가능한 Parameter 없이 Upsampling이 이루어지고, Transposed Convolution은 학습 가능한 Parameter들이 존재하고 학습을 통해 Upsampling이 이루어지는 것으로 이해하였습니다.



과제2의 경우에서, 4번의 Maxpooling 단계를 거치고 16배 작아진 해상도를 

torch.nn.Upsample(scale_factor=16, mode='bilinear', align_corners=False) 로 다시 16배 크기로 Upsampling 하는 과정 속에서 과연 제대로 추출된 feature들을 정의된 보간법 만으로 잘 활용할 수 있을지 궁금증이 생겼습니다.

또, torch.nn.Upsample과 Transposed Convolution의 성능 차이가 존재하는 것인지

그렇다면 어떤 상황에서 두 방법중 하나를 써야하는 건지, 조합해서 사용하는 건지 궁금합니다





> 말씀하신대로 단순한 interpolation만을 이용한 upsampling의 경우 정보 손실이 일어날 수 있습니다. 따라서, 보다 정확한 upsampling을 원한다면 transposed convolution과 같은 학습 가능한 upsampling을 사용하는 것이 좋습니다. 다만, 언제 반드시 어떤 방법을 사용해야만 한다는 규칙은 없고, 모델과 상황에 따라 적절한 upsampling 방법을 선택하고 실험하며 적용하는 것이 하나의 방법입니다.