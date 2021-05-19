# 경사하강법 응용 (SGD)

## 1. 경사하강법은 과연 만능인가?



![image-20210130222107799](0126_Gradient Descent(2).assets/image-20210130222107799.png)

- 위의 식이 직접 지난 시간에 유도까지 해보았던 선형회귀 계수추정을 위한 그레디언트 벡터이다.
- 하지만 이를 그대로 (L2 norm)을 그대로 사용하여 해당 문제를 해결하려고 하는 것보다, 이를 제곱해서 최소화 문제를 푸는 것 역시 동치이기 때문에 이것이 더 연산을 수월하게 한다.

![image-20210130222129850](0126_Gradient Descent(2).assets/image-20210130222129850.png)

- `경사하강법`은 `무어펜로즈역행렬`을 이용하는 것처럼 수치를 정확하게 구해주는 것이 아니고 반복을 통해 해를 서서히 찾아 수렴하는 알고리즘이다.
- 만약 학습횟수(`epoch`)와 학습률(`LearningRate`)을 적절히 설정하지 못하면  정답을 찾지 못하는 경우가 생길 수가 있다.
- 이와 같이 경사하강법은 만능은 아니다. 그저 역행렬을 사용하지 않고도 최적화가 가능한 알고리즘이라 편한 것.
- 이론적으로 경사하강법은 미분가능하고 볼록(`Convex`) 함수에 대해선 적절한 학습률과 학습횟수를 선택했을 때엔 수렴이 보장되어 있긴 합니다.

![image-20210130222343436](0126_Gradient Descent(2).assets/image-20210130222343436.png)



- 선형회귀의 경우에는 목적식 `L2 norm`이 회귀계수 *Beta*에 대해 볼록함수이기 때문에 알고리즘을 충분히 돌리면 수렴이 가능하다는 것이 저명한 것입니다.
  - 하지만 다음 그림과 같은 비선형회귀 문제를 해결할 때에는 `NonConvex`함수를 다뤄야하기 때문에 이를 위한 방법이 필요하다. (=> SGD)

![image-20210130222458100](0126_Gradient Descent(2).assets/image-20210130222458100.png)



![image-20210130222436674](0126_Gradient Descent(2).assets/image-20210130222436674.png)



# SGD가 뭔데?

> **SGD는 모든 데이터에 대해서 그레디언트를 계산하는 것이 아니라 일부의 데이터만을 사용해 그레디언트를 계산하고 업데이트 하는 과정을 수행한다는 점에서 경사하강법과 다르다**



![image-20210130222922968](0126_Gradient Descent(2).assets/image-20210130222922968.png)

- 이렇게 일부의 데이터만을 가지고 `그레디언트 벡터의 추정량`을 활용해 업데이트하는 것이지만, 그 추정량은 기댓값 측면에서 보면 상당히 근사하기 때문에 이런 원리가 가능한 것이다.



## SGD의 장점

- SGD라고 해서 만능이라는 것은 아니지만 딥러닝과 같은 비선형 문제 해결의 경우 SGD가 경사하강법보다 실증적으로 더 낫다고 검증되어있다.

- SGD는 데이터의 일부만을 가지고 그레디언트 벡터를 계산하고 파라미터를 업데이트하기 때문에 연산자원을 좀더 효율적으로 활용할 수 있다는 이점이 있다.

  - 총 n개의 데이터가 있고 SGD의 배치사이즈가 d라고 한다면 한다면, 경사하강법은 n^b만큼 연산량을 가진다.
  - 하지만, SGD는 d^b만 연산량을 가진다고 생각할 수 있겠다. 

  

  ![image-20210130223308225](0126_Gradient Descent(2).assets/image-20210130223308225.png)

![image-20210130223444767](0126_Gradient Descent(2).assets/image-20210130223444767.png)



- SGD는 미니배치를 가지고 그레디언트를 계산하는데, 미니배치는 확률적으로 선택되므로 목적식 모양이 바뀌게 됩니다.

![image-20210130224011293](0126_Gradient Descent(2).assets/image-20210130224011293.png)



- 목적식의 모양이 바뀐다는 것은, 서로 다른 미니배치를 사용하기 때문에 **`곡선모양이 바뀌고, 원래의 극소점이 이후 시점에선 극소점이 아니게 된다`**는 것이다.
- 만약, `Non-Convex`인 경우에 `Local Point`에(극소점이나 극대점에) 도착해버린 경우에도 `SGD`는 확률적으로 목적식이 바뀌게 되기 때문에 해당 지점이 더이상 Local Point가 아니게 할 수 있는 확률이 있다.
- 따라서, (본래 Gradient Descent에서는 local point에 빠져 탈출이 불가능한 경우였다 하더라도)  SGD는 탈출을 가능하게 할 확률이 존재하게 된다.
  - 사실 `local minimum`에 빠졌을 때, 무조건 탈출한다고 볼 수도 없고, 오히려 `global minimum`에서도 탈출하게 되는 경우가 생길 수도 있겠다.
  - 그저 배치를 매 `epoch`마다 random하게 사용해서 업데이트할 수 있다보니 한번 빠지게 된 `local minimum`이 다음 배치훈련 때는 `local minimum`이 아니게 된다는 단순한 이론인 것이다.



![image-20210130224219049](0126_Gradient Descent(2).assets/image-20210130224219049.png)

- 주의할 점 : `SGD`에서 미니배치 사이즈를 너무 작게 잡게 되면 오히려 경사하강법보다 수렴속도가 느려질 수 있다.
- 따라서 배치사이즈를 적절하게 잡아서 `SGD` 알고리즘을 사용하여야 한다.



> 이러한 목적식이 계속 바뀌는 `SGD`의 특성상, 등고선 위에서 보자면 수렴하는 모습이 다음과 같다.
>
> - 전체데이터가 아니라 일부분 (미니배치)만을 가지고 그레디언트 벡터를 계산하기 때문에 부분부분의 각 화살표를 진행하는데 걸리는 시간이 훨씬 적게 걸린다 (같은 시간 대비 더 많이 움직일 수 있다.)
>

![image-20210130224246472](0126_Gradient Descent(2).assets/image-20210130224246472.png)



> **추가적으로 궁금했던 점**

그럼 예시를 들어보자.

`데이터가 10만개`이고 `batchsize=1만` 으로 정했다면,

한 epoch에서 1만개씩 각자의 나눠진 배치에 따라 10번의 (`그레디언트벡터 계산`, `파라미터 업데이트`) 모델학습을 진행한다는 것까진 알겠다.

**1번 질문**

​	1) 그럼 학습과정 중에 1번의 epoch 내에서의 10번 미니배치학습 중 서로 목적함수(`loss function`)값을 비교하여 최고의 배치학습 1개를 고르고, 해당 파라미터가 해당 epoch을 마칠 때 업데이트되는 파라미터가 되는 걸까? (=> 이게 맞다면 거의 `cross validation`과 개념이 유사해져버리는듯.. 이거 아닐듯)

​	2) 아니면 1번의 epoch당 정말 10번의 `학습`이 이루어지는 것 (이렇다면, 총 학습 횟수는 `epoch*batchgroup`)이고, 하나의 epoch 내에서 이 10번의 학습끼리 이전 배치학습에서 업데이트된 파라미터가 다음 파라미터 업데이트 과정에 영향을 미치는 초기값이 되는 것이며 결국 해당 epoch에서의 최종 파라미터 업데이트 값은 10번 중 마지막 10번째 배치학습의 결과 파라미터 값이 되는 것일까?



**2번 질문**

SGD는 계속해서 구하는 목적함수가 다르게 되면 global minimum을 찾았는데도 이후 배치학습에서 탈출해버리는 경우가 생기지 않을까?

정답) 맞음. 이후 뽑히는 배치로 생성된 목적식의 minimum이 local minimum일 수도 있기 때문



추가설명 +) 사실 SGD의 큰 단점중 하나가, global minimum 근처까지는 빠르게 수렴하지만 정작 그 근처에서는 global minimum에 도달하지 못하고 진동하는 경우인데 이게 미니배치를 통해 매번 목적식을 새로 구하다보니 발생하는 문제거든요. 이에 반해서 GD의 경우 global minimum까지 가는데 필요한 연산량, 시간이 많지만 전체 배치를 통해 모든 점에서 일정한 목적식을 사용하다보니 convex function의 경우 global minimum으로 확실히 수렴할 수 있는 거구요.

SGD의 위와 같은 문제점은 learning rate를 점점 줄여나가는 방식을 사용하게 되면, global minimum 근처에서 step size가 그만큼 많이 줄어들어 크게 벗어나지 않는 수준으로 어느정도 해결할 수 있을 것 같다고 합니다.