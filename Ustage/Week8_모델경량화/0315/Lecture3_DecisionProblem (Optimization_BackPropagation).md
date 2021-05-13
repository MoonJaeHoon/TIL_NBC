## **1. 결정 (Decision)**

 

- **연역적 결정 (연역적 논리) : Defnition(정리) -> Theorem(증명)

  **전제가 참이면 결론이 무조건 참



![img](Lecture3_DecisionProblem%20(Optimization_BackPropagation).assets/img1.png)

 

- **귀납적 결정 (귀납적 논리) : 개별적인 특수한 사실이나 현상(항상 참이 아닐 수 있음)에서 그러한 사례들이 포함되는 일반적인 결론을 이끌어내는 추론 형식의 추리 방법**

 

 

 

## **2. 결정기 (Decision making machine)**

 

**결정기 (Decision making machine)**는 **원래 사람이 내려야 하는 결정 부분을 기계가 대신해주는 것**을 의미합니다.

즉, **데이터 기반으로 의사결정을 내리는 지원 시스템**이며, **머신러닝 모델**을 의미합니다.

 

- **가장 기본적인 결정기 (모델) : "평균"**



![img](Lecture3_DecisionProblem%20(Optimization_BackPropagation).assets/img2.png)



 

 

- **최근 결정기 : "다양한 모델"**



![img](Lecture3_DecisionProblem%20(Optimization_BackPropagation).assets/img3.png)



 

 

 

## **3. 가벼운 결정기 (Lightweight decision making machine)**

 

일단, **경량화(Lightweight)**와 **소형화 (Miniaturization)**의 차이는 다음과 같습니다.

 

- **경량화**는 모델의 규모 등이 이전보다 줄거나 가벼워지는 것을 뜻합니다.
- **소형화**는 사물의 형체나 규모가 작아지는 것을 뜻합니다. 

 

#### **1) TinyML**

TinyML은 **Mobile AI보다 훨씬 경량화시키는 모델**을 의미합니다.

 



![img](Lecture3_DecisionProblem%20(Optimization_BackPropagation).assets/img4.png)

 

 

#### **2) Edge intelligence**

Cloud intelligaence에서는 중앙 집권적으로 정보를 처리합니다. 하지만 이렇게 되면 중앙 서버에서 과부하 문제가 생길 수 있습니다.

Edge intelligence에서는 중앙 서버가 일을 하지만 간혈적으로 일을 처리합니다. **많은 데이터들은 Edge에서 일을 처리하거나 Edge단에 있는 서버에 보내서 처리를 합니다.**

 

 



![img](Lecture3_DecisionProblem%20(Optimization_BackPropagation).assets/img5.png)

 

Edge Intelligence를 살펴보면 다음과 같습니다.

 

- **Edge Training :** Edge단에서 model을 training하는 것

  아직까지 상용화되지는 않았고, 연구용으로 활발히 진행 중 (저전력 문제 등)

  

- **Edge Inference :** Edge단에서 model이 추론하는 것

  많이 상용화되어가고 있는 중

  

- **Edge Caching :** model이 Inference할 때 필요한 데이터들을 CPU, GPU 근처에 두고 처리하는 방법

  Edge에서 처리하기가 버겁지만 Cloud까지 보내기에는 애매한 데이터들을 처리

  

- **Edge Offloading :** Edge 서버, 즉, Cloud 서버는 아니지만 Edge 근처의 서버를 말함

  데이터를 offloading해서 필요할 때마다 가져다 쓰는 개념

 



![img](Lecture3_DecisionProblem%20(Optimization_BackPropagation).assets/img6.png)

 

 

현재 Edge Inference를 depth 있게 보면 다음과 같습니다.

 

High level에서 사용하는 PyTorch나 Tensorflow 등으로 Model을 대부분 만들게 되는데 이 것들이 Edge devices로 내려가기 위해서는 여러 가지 과정을 거쳐야 합니다.

 

High level IR에서 Low level IR로 **Graph lowering**을 시킨다고 합니다.

즉, low level 언어로 compile을 하게 됩니다.

 



![img](Lecture3_DecisionProblem%20(Optimization_BackPropagation).assets/img7.png)

 

 

## **4. Optimizer**

#### **1) 연역적 optimize vs 귀납적 optimize**

 

- **연역적 결정 (연역적 논리)**

**컴퓨터가 문제를 푸는 과정**이라고 말할 수 있습니다.

 

컴퓨터가 최적화된 답을 내놓을 때까지의 과정은 **모든 가능한 combination을 고려**해가면서 update 하는 과정을 거칩니다. 따라서 오른쪽 연두색 선과 같이 모든 가능성을 고려하기 전까지는 optimal을 고정할 수 없습니다.

 



![img](Lecture3_DecisionProblem%20(Optimization_BackPropagation).assets/img8.gif)

 

 

- **귀납적 결정 (귀납적 논리)** 

**머신러닝이 문제를 푸는 과정**이라고 말할 수 있습니다.

 

머신러닝이 optimizer를 자기 스스로 하는 과정에서는 모든 가능한 combination을 고려하는 것이 아니라 각 순간마다 optimizer를 수행하게 됩니다.

 



![img](Lecture3_DecisionProblem%20(Optimization_BackPropagation).assets/img9.gif)

 

 

#### **2) Decision problem vs optimization problem**

예시로 **Decision Spanning Tree (DST)**와 **Minimum Spanning Tree(MST)**를 비교해봅시다.

 

**Decision problem은 Decision Spanning Tree (DST) 문제입니다.**

아래 Graph G에 weight가 주어졌을 때 spanning tree(모든 노드들을 이으면서 cycle이 만들어지지 않는 상태)를 가지는지 물어보는 문제입니다.

따라서 이 weight cost의 합이 사전에 정해놓은 kk라는 값보다 작은 지를 물어보는 문제입니다.

 

**Decision problem은 Cost의 upper bound가 정해져 있고, 그 upper bound만 만족하면 풀리는 문제입니다.**

 



![img](Lecture3_DecisionProblem%20(Optimization_BackPropagation).assets/img10.png)

 

**Optimization problem은 Minimum Span\**n\**ing Tree(MST) 문제입니다.문제입니다.**

 

똑같이 Graph G에 weight가 주어졌을 때 Cost를 점점 내리면서 Dicision problem을 반복합니다. 이때, 더 이상 내리지 못하는 Cost의 값을 찾는 문제입니다.

 

즉, **Optimization problem은 Dicision problem을 연쇄적으로 반복했을 때 해결할 수 있습니다.**

 

 

 

#### **3) Optimization problem in DL**

**DL에서 Decision problem는 Inference를 뜻합니다.**

nerual network가 주어졌을 때 "validation loss의 upper bound" kk 안에 포함되는 neural network를 Inference 하는 문제입니다.

 

**DL에서 Optimization Problem은 minimize 된 loss를 Inference 하는 것을 뜻합니다.**

Decision problem에서 validation loss kk의 값을 계속 줄여가면서 validation loss가 minimize 하는 Inference를 찾는 것을 뜻합니다.

 

 

 

## **5. Constraints**

 

**Constraints**는 사용자가 정해놓은 제약사항을 뜻합니다. 

 

일반적으로 DL에서 performance를 높이기 위해서는 Cost를 지불하게 됩니다.

 

**Decision problem**에서 Performance를 높이기 위해 무제한의 자원 (infinite amount of resoources)을 사용한다고 가정합니다.

 

**Optimization problem**에서는 각각의 Decision problem에서 Performance를 높이기 위해 지불했던 Cost를 더해서 Constraints를 계산합니다. 이때, **모든 Cost를 더했을 때 Constraint를 넘으면 안 됩니다.**

 



![img](Lecture3_DecisionProblem%20(Optimization_BackPropagation).assets/img11.png)