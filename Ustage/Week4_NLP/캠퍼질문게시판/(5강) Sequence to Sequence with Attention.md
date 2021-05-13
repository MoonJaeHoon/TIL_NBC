## 1. Teacher Forcing에 대한 질문이 있습니다.

Teacher Forcing을 사용하는 경우에는 매 time step별로 Ground Truth 값을 주어 학습이 가능한데,

이를 적용하지 않는 경우에는 어떤 방식으로 해당 출력의 loss를 계산하게 되나요? 



> Teacher Forcing을 사용하지 않는 경우라고 해도 매 time step의 input으로 ground truth를 주지 않을 뿐, 당연히 학습 과정에서 저희는 ground truth를 알고 있으므로 그냥 label로 똑같이 loss를 계산해주면 됩니다. 아예 ground truth를 모르는 inference 과정이 아닌 이상 loss를 구하는 과정을 똑같습니다.