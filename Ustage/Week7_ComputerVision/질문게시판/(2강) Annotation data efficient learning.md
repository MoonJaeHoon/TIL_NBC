# 1. 준지도학습 (semi-supervisied learning) 관련 질문있습니다

1.  self-learning에서 iteration에 따라서 지속적으로 teacher model을 통해 unlabeled-data를 pseudo-labeled-data로 바꿔서 원래 labeled-data와 함께 student model이 학습을 하는 것이라고 하셨는데, 그렇다면 굳이 iteration이 필요한지 의문입니다. 왜냐하면 어차피 다음 iteration에서 또 unlabeld된 데이터가 들어오고, pseudo-labeling을 할 텐데, 그것이 곧, 이전 iteration에서 student model(지금 시점에서는 teacher model)이 학습한 결과와 같기 때문이라고 생각합니다. 혹시 매 iteration마다 들어오는 진짜 label된 데이터의 label 또한 달라지기 때문에 반복의 여지가 있는건가요? 아니라면 어디서 반복의 여지가 있는건지 궁금합니다.



> 말씀해주신 방법대로 진행하는 semi-supervised learning 방법을 pseudo-labeling 혹은 self-training이라고 합니다. 
>
> 그 중 최근에 주목받은 방법이 구글에서 발표한 noisy student method (https://arxiv.org/abs/1911.04252)입니다. 
>
> 해당 논문에서도 강조하기를 매 iteration마다 새로운 student model을 labeled data + pseduo-labeled data에 대해 학습을 진행할 때, 3가지 noise(data augmentation, dropout, stochastic depth)를 추가하는 것이 굉장히 중요하다고 합니다. 
>
> 결국 이러한 noise가 추가된 student model이 pseudo-labeled data에 학습이 진행되면 기존의 결과와 앙상블된 효과를 얻게 되고 최종적으로 성능이 매우 뛰어난 모델을 얻을 수 있게 됩니다 :)



1-2. 빠른 답변 감사합니다. 그런데 제가 똑바로 이해했는지 모르겠습니다. 결국, 1번째 iteration에서 노이즈를 줘서 나온 모델 a보다 2번째 iteration에서 노이즈를 줘서 나온 모델 b는 이미 a에서 1번째 iteration에서 data-augmentation, dropout, stocastic depth를 거친 모델에서 또 3가지 행위를 한다는 점에서 성능 개선의 여지가 있기에 계속 반복을 돌리는 건가요?



> 논문에서 해당 부분을 기술하고 있는 문장을 아래에 첨부하였습니다. 정리하면 말씀드린 3가지 noise를 통해서 적은 iteration을 돌려서 만든 teacher network보다 새로운 noise를 추가하여 학습이 된 student network가 더욱 성능이 뛰어날 것이라는 가정이었고 실제로도 성능 개선을 보인 모델이라고 할 수 있을 것 같습니다 :)
>
>  
>
> During the learning of the student, we inject noise such as dropout, stochastic depth, and data augmentation via RandAugment to the student so that the student generalizes better than the teacher





# 2. Pseudo Label : 일정 threshold값을 기준으로

pseudo label을 사용하는 것에 대해서 질문입니다.

labeled data로 충분히 학습이된 모델에서 unlabeled data가 일정 threshold를 넘는 softmax값을 가졌을 때, pseudo label로 사용하나요? 아니면, 그냥 모든 전체 unlabled data를 pseudo로 사용하나요?



> unlabeled data의 경우 confidence value가 0.3 이상인 데이터만 pseudo label data로 사용하게 됩니다.