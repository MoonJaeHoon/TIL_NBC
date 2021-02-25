## 그래프에서의 군집(Community)

통계에서의 군집화(Clustering)와 다른 점:

통계에서의 Clustering은 어떤 인스턴스 벡터들에 대하여 군집화하는 알고리즘을 뜻하지만, 그래프에서의 군집화는 어떤 정점(노드)에 대해서 군집을 발견해내는 알고리즘을 뜻합니다.



그래프에서는 군집탐색의 성공여부를 판단하기 위해 군집성(Modularity, 모듈성)을 사용합니다.



### 1.1 Girvan-Newman



### 1.2 Louvain







### ※ 예외 : 통계에서의 군집화(Clustering)를 그래프 Community 구성에 활용하는 방법도 있다.



> 이 경우에는 그래프의 노드(정점)을 인스턴스 벡터화시켜주는(임베딩해주는) Node2Vec과 같은 알고리즘을 적용한 뒤 해당 Embedding Vector에 K-means와 같은 Clustering을 적용하게 된다.

