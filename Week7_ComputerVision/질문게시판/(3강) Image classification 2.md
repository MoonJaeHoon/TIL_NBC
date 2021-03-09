# 1. CNN에서의 feature에 대해 질문있습니다.

NLP에서 feature vector는 임베딩이라 생각했었는데요,

CNN에서는 feature가 '거쳐온 filter들'에 해당하는지 궁금합니다.



> CNN에서 feature는 broad한 의미인데, 보통 filter를 거친 feature map을 이미지의 특성을 나타내는 feature라고 볼 수 있습니다.





# 2. GoogLeNet에서 Auxiliary classifier관련 질문드립니다.

다음 사진에서 Conv와 MaxPool, AvgPool에 filter size(1*1, 3*3, 5*5) 옆에 +1(S), +3(V)는 어떤 의미인지 알 수 있을까요?[<img src="(3강) Image classification 2.assets/IMG_738E33B08495-1.jpeg" alt="img" style="zoom: 50%;" />





> 모두 stride를 의미하는 겁니다~!!





# 3. Resnet architecture관련 질문드립니다.

feature map size가 절반으로 되는 경우에 time complexity per layer를 유지하기 위해 filter의 수를 2배로 늘린다는 점에서 의문이 생겼습니다!

1. time complexity per layer를 유지하는 이유가 궁금합니다!

2. 만약 논문에서처럼 time complexity per layer를 유지한다고 했을때, 다음과 같은 의문이 들었습니다.

"feature map size를 절반으로 하는 것은 feature map의 가로 세로 모두를 절반으로 하는 것이고, 여기에서 filter의 수를 2배로 늘리면 최종적으로 연산횟수는 1/2로 줄어드는 것인데 이것을 time complexity가 줄어든다고 표현하는게 맞는 것인가?" 아니면 처음부터 접근을 잘못한 것인지 궁금합니다(접근을 잘못한 것이라면 어떻게 이해를 하는게 맞는 것인지가 궁금합니다!)



출처 : https://www.edwith.org/bcaitech1/forum/53787