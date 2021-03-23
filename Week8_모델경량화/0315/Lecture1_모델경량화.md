이 글은 아래와 같은 내용으로 구성된다.

- Edge Device
  - [Edge Intelligence](https://olenmg.github.io/2021/03/15/boostcamp-day36.html#edge-intelligence)
  - [Optimization problem](https://olenmg.github.io/2021/03/15/boostcamp-day36.html#optimization-problem)
- [Backpropagation](https://olenmg.github.io/2021/03/15/boostcamp-day36.html#backpropagation)
- [Reference](https://olenmg.github.io/2021/03/15/boostcamp-day36.html#reference)



## Edge Device

모델을 경량화하는 대개의 이유는, **edge device에 모델을 넣기 위함이다.** 그렇다면 edge device는 뭘까? 그냥 우리가 흔히 들고다니는 스마트폰도 edge device의 일종이 될 수 있다. 아래 그림을 보자.

![edge_device]((1%EA%B0%95)%20%EA%B0%80%EB%B2%BC%EC%9A%B4%20%EB%AA%A8%EB%8D%B8.assets/36-1.png)
가운데에 그려져 있는 그래프는 network graph로, 인터넷으로 연결될 수 있는 각종 시스템들이 그려져있다. **Edge device**는 위와 같이 네트워크 구조에서 말단(edge)에 있는 것을 말한다.

모델 경량화는 모델을 cloud나 on-premise등 크기가 큰 곳에 원격으로 올리지 않고 edge device 자체에 올리는 한편 그 device에서 돌아갈 수 있도록 모델의 크기나 계산량을 줄이기 위해 존재한다.

그렇다면 우리는 왜 모델을 edge device에 올리려고 할까? 당연히 속도 때문이다. LAN을 통해 cloud나 여타 서버에 접속하는 latency를 줄이려면 그 모델이 직접적으로 필요한 곳에 모델을 올려야 한다. 또한 cloud 등에 올리려면 금전적 비용도 만만치 않다. 이러한 금전적 비용도 줄이기 위해 경량화가 사용된다.

edge device라는 단어를 찾아보면 ‘Dumb and fast’라는 구절이 나온다. 같지만(성능은 떨어지지만) 빠르다는 것을 의미한다.



#### Edge Intelligence

우리가 Edge device에 모델을 올린다고 하면 device 상에서 무엇을 할 수 있을까? training? inference? 당연하겠지만 지금 당장의 목표는 inference이다. training부터 edge device에서 시작하는 것은 아직 갈 길이 멀다. training에는 많은 리소스가 소모된다. 현 시점에서 모델 경량화라고 하면 대부분이 inference에 집중해있는 편이다.

![edge_intelligence]((1%EA%B0%95)%20%EA%B0%80%EB%B2%BC%EC%9A%B4%20%EB%AA%A8%EB%8D%B8.assets/36-2.png)
한편, Edge단에 가까운 서버를 두어 latency를 최소화하되 edge에 직접 모델을 다 올리지는 않는 **edge offloading**이나 training/inference 시에 필요한 데이터를 미리 캐싱해놓는(hit를 최대화하고, miss를 최소화하는) **edge caching** 등의 literature도 존재한다.



#### Optimization problem

이건 비단 Edge device만의 문제는 아니지만 아무튼 역시 관련이 있는 것이기 때문에 다루어보도록 한다. 우리가 결국 하고자 하는 것은 최고의 성능을 내는 모델을 만드는 것인데, 문제는 resource가 제한적이라는 것이다. 우리는 주어진 환경(cost constraint) 내에서 최고의 성능을 내야한다. 현실은 그렇다.

그런데 그 제약조건에는 무엇이 있을까? 당연히 GPU 등 하드웨어적인 제한도 있겠지만 그건 당연하고 더 나아가 저작권, 개인정보 보호, 심지어는 이전 특강때 다뤘듯이 온실가스 배출량 등도 제약조건이 될 수 있다.

지금, 배우는 단계에서는 제약이라해봐야 하드웨어 리소스 제약, 조금 더 넓게 보면 저작권 제약 정도밖에 고려하지 못한다. 하지만 추후를 생각해서라도 항상 다양한 제약조건에 대해 고민해보고 이들을 만족하는 모델을 설계할 수 있어야 한다.



## Backpropagation

![NN]((1%EA%B0%95)%20%EA%B0%80%EB%B2%BC%EC%9A%B4%20%EB%AA%A8%EB%8D%B8.assets/36-3.png)
위와 같은 간단한 linear layer들로 이루어진 신경망이 있다고 해보자. binary classification을 수행한다고 가정하면, 마지막 output에도 sigmoid를 적용할 수 있다. input xx에 대한 forward pass는 아래와 같다.

y1=x⋅W1z1=σ(y1)y2=z1⋅W2^y=σ(y2)L=∥y−^y∥22(1)(2)(3)(4)(5)(1)y1=x⋅W1(2)z1=σ(y1)(3)y2=z1⋅W2(4)y^=σ(y2)(5)L=‖y−y^‖22

역전파를 통해 구해야하는 식은 다음과 같다.

∂L∂W2=∂L∂^y∂^y∂y2∂y2∂W2∂L∂W2=∂L∂y^∂y^∂y2∂y2∂W2∂L∂W1=∂L∂^y∂^y∂y2∂y2∂z1∂z1∂y1∂y1∂W1∂L∂W1=∂L∂y^∂y^∂y2∂y2∂z1∂z1∂y1∂y1∂W1

지금은 W2W2에 대한 역전파 값만 구해보도록 한다.

먼저 batch가 1일 때를 생각해보면, 각 미분값은 아래와 같다.

∂L∂^y=2⋅(y−^y)∂L∂y^=2⋅(y−y^)∂^y∂y2=σ(y2)⋅(1−σ(y2))∂y^∂y2=σ(y2)⋅(1−σ(y2))∂y2∂W2=z⊺1∂y2∂W2=z1⊺

여기서 주의해야 할 식은 가장 아래 식인데, 보다시피 transpose가 적용된 값이 나오게 된다. 한 가지 더 주의할 점은, 여기서 **식의 계산순서가 정해진다는 점이다.** 이를 식으로 표현하면 아래와 같다.

∂y2∂W2=z⊺1⋅∂L∂y2∂y2∂W2=z1⊺⋅∂L∂y2

한편, 두 번째 식의 의미는 sigmoid 함수 σ(x)σ(x)에 대한 미분이 σ(x)(1−σ(x))σ(x)(1−σ(x))라는 점을 상기하면 쉽게 알 수 있다.

이러한 점들에 주의하여 backpropagation을 from scratch로서 구현하면
`d_weight2 = np.dot(z_1.T, 2 * (y_hat - self.y)) * ((1 - y_hat) * y_hat)`
와 같게 된다.

위 설명에는 명시하지 않았지만 원래 affine layer에는 bias 값도 존재한다. 그런데 이것을 실제 구현할 때는 따로 명시하지 않고 affine layer의 가중치 값에 b 값도 함께 넣는 한편 **해당 부분과 곱해지는 input 부분을 고정으로 1로 주면** 쉽게 구현할 수 있다. 다만 bias 부분의 경우 backpropagation 시 차원이 맞지 않으므로 해당 axis로 `sum`을 적용 후 가중치를 업데이트해주어야 한다.

여기서 가장 중요한 것은 개인적으로 **affine layer의 미분**이라고 생각한다. 솔직히 아직 확실히 이해하지는 못했지만, 계산 순서가 정해진다는 점이나 transpose가 된 값이 미분결과로 튀어나온다는 점을 기억해야 할 것이다.

한편, 위에서 썼던 코드는 batch size가 1이든 1보다 크든 잘 돌아간다. 왜일까? 이를 알기 위해 input이 (1, 1)에서 (4, 1)로 바뀔 때 고려해야 할 점을 생각해보자.

내가 가장 먼저 들었던 생각은 배치가 늘어나면 weight을 여러 번 업데이트해주어야 하는데 그것이 어떻게 되느냐였다. 직접 써보면 결국 **행렬 곱에 의해, 각 배치마다 업데이트해야하는 값이 한 번에 다 더해진다.** 즉, batch 단위로 신경망을 통과시킬 경우 batch 단위로 한 번에 레이어의 가중치 값을 업데이트할 수 있다.

batch size가 1일 때는 한 번에 업데이트하는 구간이 없으므로 아무래도 batch size가 1보다 클 때보다 업데이트가 보다 섬세하게(조금씩) 이루어질 것이다. 다만 병렬연산을 활용하지 않으므로 시간이 오래 걸릴 것이다.

batch를 늘렸을 때 forward/backward에 관여하는 행렬곱 연산들이 모두 가능한지도 생각해보아야한다. 하지만 여기서 forward pass의 특성상 input의 첫 번째 shape은 불변하므로 중요한 것은 두 번째 shape이다. batch는 첫 번째 input의 첫 번째 shape으로 들어오게 되므로 계산에 무리가 없다. forward pass에 문제가 없다는 것은 backward pass에서도 shape에 문제가 없다고 이해할 수 있다.

위에서 기술한 각종 의문점들은 batch가 1일 때와 1보다 클 때 각각에 대하여 backpropagation에 관여하는 모든 변수의 shape을 써보면 이해할 수 있다.