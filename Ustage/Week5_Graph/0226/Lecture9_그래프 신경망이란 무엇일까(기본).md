#### 1. 정점 표현 학습 복습

1.1 정점 표현 학습

- 정점 표현 학습이란 그래프의 정점들을 벡터의 형태로 표현하는 것입니다
- 정점 표현 학습은 간단히 정점 임베딩(Node Embedding)이라고도 부릅니다
- 정점 표현 학습의 입력은 그래프입니다
- 주어진 그래프의 각 정점 𝑢에 대한 임베딩, 즉 벡터 표현 𝒛𝑢가 정점 임베딩의 출력입니다
- 그래프에서의 정점간 유사도를 임베딩 공간에서도 “보존”하는 것을 목표로 합니다
- 그래프에서 두 정점의 유사도는 어떻게 정의할까요?
  - 그래프에서 정점의 유사도를 정의하는 방법에 따라, 인접성/거리/경로/중첩/임의보행 기반 접근법으로 나뉩니다

1.2 변환식 정점 표현 학습과 귀납식 정점 표현 학습

- 지금까지 소개한 정점 임베딩 방법들을 변환식(Transductive) 방법입니다
- 변환식(Transdctive) 방법은 학습의 결과로 정점의 임베딩 자체를 얻는다는 특성이 있습니다
- 정점을 임베딩으로 변화시키는 함수, 즉 인코더를 얻는 귀납식(Inductive) 방법과 대조됩니다
- 출력으로 임베딩 자체를 얻는 변환식 임베딩 방법은 여러 한계를 갖습니다
  1) 학습이 진행된 이후에 추가된 정점에 대해서는 임베딩을 얻을 수 없습니다
  2) 모든 정점에 대한 임베딩을 미리 계산하여 저장해두어야 합니다
  3) 정점이 속성(Attribute) 정보를 가진 경우에 이를 활용할 수 없습니다
- 출력으로 인코더를 얻는 귀납식 임베딩 방법은 여러 장점을 갖습니다
  1) 학습이 진행된 이후에 추가된 정점에 대해서도 임베딩을 얻을 수 있습니다
  2) 모든 정점에 대한 임베딩을 미리 계산하여 저장해둘 필요가 없습니다
  3) 정점이 속성(Attribute) 정보를 가진 경우에 이를 활용할 수 있습니다

#### 2. 그래프 신경망 기본

2.1 그래프 신경망 구조

- 그래프 신경망은 그래프와 정점의 속성 정보를 입력으로 받습니다
- 그래프의 인접 행렬을 A라고 합시다
- 인접 행렬 A은 |𝑉|×|𝑉|의 이진 행렬입니다
- 각 정점 𝑢의 속성(Attribute) 벡터를 𝑋𝑢라고 합시다
- 정점 속성 벡터 𝑋𝑢는 𝑚차원 벡터이고, 𝑚은 속성의 수를 의미합니다
- 정점의 속성의 예시는 다음과 같습니다
  - 온라인 소셜 네트워크에서 사용자의 지역, 성별, 연령, 프로필 사진 등
  - 논문 인용 그래프에서 논문에 사용된 키워드에 대한 원-핫 벡터
  - PageRank 등의 정점 중심성, 군집 계수(Clustering Coefficient) 등
- 그래프 신경망은 이웃 정점들의 정보를 집계하는 과정을 반복하여 임베딩을 얻습니다
- 예시에서 대상 정점의 임베딩을 얻기 위해 이웃들 그리고 이웃의 이웃들의 정보를 집계합니다
- 각 집계 단계를 층(Layer)이라고 부르고, 각 층마다 임베딩을 얻습니다
- 각 층에서는 이웃들의 이전 층 임베딩을 집계하여 새로운 임베딩을 얻습니다
- 0번 층, 즉 입력 층의 임베딩으로는 정점의 속성 벡터를 사용합니다
- 대상 정점 마다 집계되는 정보가 상이합니다
- 대상 정점 별 집계되는 구조를 계산 그래프(Computation Graph)라고 부릅니다
- 서로 다른 대상 정점간에도 층 별 집계 함수는 공유합니다
- 서로 다른 구조의 계산 그래프를 처리하기 위해서는 어떤 형태의 집계 함수가 필요할까요?
- 집계 함수는 (1) 이웃들 정보의 평균을 계산하고 (2) 신경망에 적용하는 단계를 거칩니다
- 마지막 층에서의 정점 별 임베딩이 해당 정점의 출력 임베딩입니다

2.2 그래프 신경망의 학습

- 그래프 신경망의 학습 변수(Trainable Parameter)는 층 별 신경망의 가중치입니다
- 먼저 손실함수를 결정합니다. 정점간 거리를 “보존”하는 것을 목표로 할 수 있습니다
- 변환식 정점 임베딩에서처럼 그래프에서의 정점간 거리를 “보존”하는 것을 목표로 할 수 있습니다
- 만약, 인접성을 기반으로 유사도를 정의한다면, 손실 함수는 다음과 같습니다
  ![img](https://media.vlpt.us/images/skaurl/post/c62ca7d9-50a0-45a3-a9ef-f64b6f8efc78/%EC%8A%A4%ED%81%AC%EB%A6%B0%EC%83%B7%202021-02-26%2009.41.10.png)
- 후속 과제(Downstream Task)의 손실함수를 이용한 종단종(End-to-End) 학습도 가능합니다
- 정점 분류가 최종 목표인 경우를 생각해봅시다
- 예를 들어,
  (1) 그래프 신경망을 이용하여 정점의 임베딩을 얻고
  (2) 이를 분류기(Classifier)의 입력으로 사용하여
  (3) 각 정점의 유형을 분류하려고 합니다
- 이 경우 분류기의 손실함수, 예를 들어 교차 엔트로피(Cross Entropy)를, 전체 프로세스의 손실함수로 사용하여 종단종(End-to-End) 학습을 할 수 있습니다
  ![img](https://media.vlpt.us/images/skaurl/post/4ed0e5f6-2360-43e1-bb43-dce6f77ef64a/%EC%8A%A4%ED%81%AC%EB%A6%B0%EC%83%B7%202021-02-26%2009.42.26.png)
- 그래프 신경망과 변환적 정점 임베딩을 이용한 정점 분류
  - 그래프 신경망의 종단종(End-to-End) 학습을 통한 분류는, 변환적 정점 임베딩 이후에 별도의 분류기를 학습하는 것보다 정확도가 대체로 높습니다
- 학습에 사용할 대상 정점을 결정하여 학습 데이터를 구성합니다
- 선택한 대상 정점들에 대한 계산 그래프를 구성합니다
- 마지막으로 오차역전파(Backpropagation)을 통해 손실함수를 최소화합니다
- 구체적으로, 오차역전파를 통해 신경망의 학습 변수들을 학습합니다

2.3 그래프 신경망의 활용

- 학습된 신경망을 적용하여, 학습에 사용되지 않은 정점의 임베딩을 얻을 수 있습니다
- 마찬가지로, 학습 이후에 추가된 정점의 임베딩도 얻을 수 있습니다
- 학습된 그래프 신경망을, 새로운 그래프에 적용할 수도 있습니다

#### 3. 그래프 신경망 변형

3.1 그래프 합성곱 신경망

- 소개한 것 이외에도 다양한 형태의 집계 함수를 사용할 수 있습니다
- 그래프 합성곱 신경망(Graph Convolutional Network, GCN)의 집계 함수입니다
- 기존의 집계 함수와 비교하여 작은 차이지만 큰 성능의 향상으로 이어지기도 합니다

3.2 GraphSAGE

- GraphSAGE의 집계 함수입니다
- 이웃들의 임베딩을 AGG 함수를 이용해 합친 후, 자신의 임베딩과 연결(Concatenation)하는 점이 독특합니다
- AGG 함수로는 평균, 풀링, LSTM 등이 사용될 수 있습니다

#### 4. 합성곱 신경망과의 비교

4.1 합성곱 신경망과 그래프 신경망의 유사성

- 합성곱 신경망과 그래프 신경망은 모두 이웃의 정보를 집계하는 과정을 반복합니다
- 구체적으로, 합성곱 신경망은 이웃 픽셀의 정보를 집계하는 과정을 반복합니다

4.2 합성곱 신경망과 그래프 신경망의 차이

- 합성곱 신경망에서는 이웃의 수가 균일하지만, 그래프 신경망에서는 아닙니다
- 그래프 신경망에서는 정점 별로 집계하는 이웃의 수가 다릅니다
- 그래프의 인접 행렬에 합성곱 신경망을 적용하면 효과적일까요?
- 그래프에는 합성곱 신경망이 아닌 그래프 신경망을 적용하여야 합니다!
- 많은 분들이 흔히 범하는 실수입니다
- 합성곱 신경망이 주로 쓰이는 이미지에서는 인접 픽셀이 유용한 정보를 담고 있을 가능성이 높습니다
- 하지만, 그래프의 인접 행렬에서의 인접 원소는 제한된 정보를 가집니다
- 특히나, 인접 행렬의 행과 열의 순서는 임의로 결정되는 경우가 많습니다

#### 9강 정리

1. 정점 표현 학습

   - 그래프의 정점들을 벡터로 표현하는 것

   - 그래프에서 정점 사이의 유사성을 계산하는 방법에 따라 여러 접근법이 구분됨

   - 그래프신경망등의귀납식정점표현학습은임베딩함수를출력으로얻음

2. 그래프 신경망 기본

   - 그래프 신경망은 이웃 정점들의 정보를 집계하는 과정을 반복하여 임베딩을 얻음

   - 후속 과제의 손실함수를 사용해 종단종 학습이 가능함

   - 학습된 그래프 신경망을 학습에서 제외된 정점, 새롭게 추가된 정점, 새로운 그래프에 적용 가능

3. 그래프 신경망 변형

4. 합성곱 신경망과의 비교

   - 그래프 형태의 데이터에는 합성곱 신경망이 아닌 그래프 신경망을 사용해야 효과적