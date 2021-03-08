## 1. RNN/LSTM/GRU 기반의 Language Model에서 초반 time step의 정보를 전달하기 어려운 점을 완화할 수 있는 방법이 있을까요?

이에 대한 답으로 저희 조에서 생각한 아이디어는 다음과 같습니다.

**"ResNet이나 DenseNet에서 이전 정보를 전달하듯이, RNN 계열의 모델에서도 직전 cell의 output 뿐만 아니라 그 전 cell의 output을 전달해서 concatenation을 해주면 문제를 완화할 수 있지 않을까?"**

예를 들자면, 아래 그림과 같은 구조가 될 것입니다.

![img](https://cphinf.pstatic.net/mooc/20210216_199/1613468343533xlRH5_PNG/mceclip2.png)

이러한 방법이 효과가 있을지, 문제점은 무엇인지 궁금합니다.



> 제시해주신 접근 방법도 좋은 방법 중 하나라고 생각합니다. 유사한 technique을 CNN에 적용하는 것이 다음 논문에서도 제시된 바 있어서 참고해보시면 좋을 것 같습니다 (Densely Connected Convolutional Networks, https://arxiv.org/pdf/1608.06993.pdf) 다만 그려주신 그대로 RNN에 적용할 경우, h를 계속 concat해주게 되면 time step에 따라 hidden dimension이 증가하기 때문에 recursive 연산은 불가능할 것 같습니다.

