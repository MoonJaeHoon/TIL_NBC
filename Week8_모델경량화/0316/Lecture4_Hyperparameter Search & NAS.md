## **1. hyperparameter search**

#### **1) Grid Layout vs Random Layout**

- **Grid Layout**은 y축 learning rate, x축 Batch size 등 hyperparameter 조합을 Grid 하게 찾아 최적 조합을 찾아내는 방식

  

- **Random Layout**은 y축 learning rate, x축 Batch size 등 hyperparameter 조합을 Random하게 찾아 최적 조합을 찾아내는 방식



![img](Lecture4_Hyperparameter%20Search%20&%20NAS.assets/img1.png)

 

 

#### **2) Surrogate Model**

Surrogate Model은 ML 모델을 정의하는 **hyperparameter들을 위한 머신러닝 모델**입니다.

 

따라서 Surrogate Model은 **hyperparameter set를 input**으로 가집니다.

hyperparameter set을 ML 모델에 넣고 나오는 **loss를** **output**으로 받습니다.

 



![img](Lecture4_Hyperparameter%20Search%20&%20NAS.assets/img2.png)

 

Surrogate의 대표적인 process가 **Gaussian process**입니다.

 



![img](Lecture4_Hyperparameter%20Search%20&%20NAS.assets/img3.png)

 

**Truth function**은 **실선**으로 표현되어있으며, **배우고자 하는 머신러닝 모델(모델 = hyperparameter를 추정하는 surrogate model)**입니다. 

**Object function**은 **점선**으로 표현되어 있으며, **추정된 함수의 평균**입니다.

**Observation**은 해당 점 위치에서 모델을 상태를 관찰한 것입니다.

 

**따라서 surrogate model의 최대값을 찾는 것은 "실제 모델"에다 해당 hyperparameter를 넣었을 때 가장 좋은 성능을 낸다는 뜻입니다.**

 

아래 오른쪽 그래프를 봤을 때, 오른쪽 점에서 최대값이 나타났기 때문에,

**exploitation**은 데이터를 토대로 "**오른쪽 점 근처에 최적값이 존재할 것"이라고 예측하는** 것입니다.

**exploration**은 표준편차가 가장 큰 점, 즉, **불확실성이 가장 높은 점 근처에 최적값이 존재하는 것이라고 예측**하는 것입니다.

단, exploitation과 exploration는 **trade-off**의 관계에 있습니다.

 

이 상태에서 세로선으로 Gaussinan 분포를 그려서 가장 Variance (파란색 분포)가 넓은 곳에 **Exploration**을 수행하는 것입니다. 이 부분은 Variance function의 분포가 넓습니다.

 

Acquistiton function은 다음 observation을 찾아가는 지표이며, 값이 가장 큰 acquisition max에서 다음 observation을 수행합니다. Mean과 Variance를 둘 다 고려해서 Acquistion max를 찾은 후, 다음 Observation을 수행합니다.

 



![img](Lecture4_Hyperparameter%20Search%20&%20NAS.assets/img4.png)

 

다음 Observation (new observation)을 찾으면 Object function이 Truth function에 수렴을 하게 됩니다.

아직 Gaussian Variance (파란색 분포)가 남은 영역은 Truth function을 못 찾은 영역, Gaussian Variance (파란색 분포)가 없는 부분은 Truth function을 이미 찾은 영역을 뜻합니다.

 



![img](Lecture4_Hyperparameter%20Search%20&%20NAS.assets/img5.png)

 

다시 다음 Observation을 찾으면서 Truth function을 찾아갑니다.

 



![img](Lecture4_Hyperparameter%20Search%20&%20NAS.assets/img6.png)

 

Gaussian process는 모든 hyper parameter를 input으로 넣어 훈련시키는 방법은 아니라서 어떤 hyper parameter가 좋은 setting인지 알아보는 데는 유의미할 수 있습니다.

 

------

**✅ EI 함수**

 

EI함수는 위에서 설명한 exploitation과 exploration을 적절히 사용하도록 설계했으며, **Acquisition Function으로 사용**합니다.

 



![img](Lecture4_Hyperparameter%20Search%20&%20NAS.assets/img7.png)

 

여기서 실선은 observation 함수의 mean을 의미합니다.

x+x+는 현재까지 나온 최대값이며, x1,x2,x3x1,x2,x3 3가지 점이 observation 되어있을 때, 각 점의 **정규 분포(초록색)**를 구해주게 됩니다.

 

이후, EI 함수 공식에 μ(x),f(x+)σ(x)μ(x),f(x+)σ(x) 값을 넣고, EI(x)의 값을 구해줍니다.

(φ(·) and Φ(·) denote the PDF and CDF of the standard normal distribution respectively)

 



![img](Lecture4_Hyperparameter%20Search%20&%20NAS.assets/img8.png)

------

 

 

## **2. Neural Architecture Search (NAS)**

 

아래 그림은 (Automatic) Neural Architecture Search (NAS)이며, **여러 candidate의 모델들의 Architecture들을 알고리즘, 딥러닝 등의 모델에 넣어서 가장 좋은 성능의 Architecture를 찾아내는 방법입니다.**

 

 



![img](Lecture4_Hyperparameter%20Search%20&%20NAS.assets/img9.png)

 

여기서 **Search strategy** 즉, 다음 후보 Architecture를 고르는 전략이 중요한데,

 

- 어디와 어디를 natural connection으로 엮을 것인가,
- 어디와 어디를 fuse를 할 것인가,
- 어디와 어디를 depth-wise seperate를 할 것인가,
- 몇 개의 layer를 쌓을 것인가,

 

이러한 strategy들을 Grid로 만들어놓고, pick 해가지고 결정하게 됩니다.

 



![img](Lecture4_Hyperparameter%20Search%20&%20NAS.assets/img10.png)

 

이러한 NAS를 적용한 논문은 다음과 같습니다.

 

- **MnasNet: Platform-Aware Neural Architecture Search for Mobile

  **NAS를 Mobile edge에서 구현한 논문
  **
  **ref) [arxiv.org/pdf/1807.11626.pdf](https://arxiv.org/pdf/1807.11626.pdf)**
  **



![img](Lecture4_Hyperparameter%20Search%20&%20NAS.assets/img11.png)

 

- **PROXYLESSNAS: DIRECT NEURAL ARCHITECTURE SEARCH ON TARGET TASK AND HARDWARE

  **Proxy 개념을 사용해서 모델 architecture의 일부만 훈련시키는 방법

  ref) [arxiv.org/pdf/1812.00332.pdf%C3%AF%C2%BC%E2%80%B0](https://arxiv.org/pdf/1812.00332.pdfï¼‰)



![img](Lecture4_Hyperparameter%20Search%20&%20NAS.assets/img12.png)

 

- **ONCE-FOR-ALL: TRAIN ONE NETWORK AND SPECIALIZE IT FOR EFFICIENT DEPLOYMENT

  **여러 Device 상황에서 다양하게 사용할 수 있게 만든 architecture search

  ref) [arxiv.org/pdf/1908.09791.pdf](https://arxiv.org/pdf/1908.09791.pdf)



![img](Lecture4_Hyperparameter%20Search%20&%20NAS.assets/img13.png)

 



![img](Lecture4_Hyperparameter%20Search%20&%20NAS.assets/img14.png)