03.17 중수님의 질문

# Complexity Cost Pruning

가지치기를 할 때, 모든 subtree를 다고려해서 valiation에 잘맞는 형태를 찾기 힘들어서 cost complexity pruning은 이를 가능하게 한다는데, 수식이나 프로세스를 보면 이 방법도 사실 다 계산을 하는거 같은데 왜 이게 모든 subtree를 고려하는게 아니게 되는지 이해가 안가서요



>  디시젼 트리(Regression Tree)의 Complexity Cost Pruning 을 위한 Loss function을 예시로 보면, 다음과 같고
>
> ![image-20210317113718674](ComplexityCostPruning(디시젼트리).assets/image-20210317113718674.png)
> $$
> |T| = 해당\ subtree에서의\ 끝마디 Node 수
> \\
> R_m : m번째\ 끝마디에\ 해당하는\ 설명변수\ 공간 (설명변수\ 공간의\ 부분집합)
> \\
> \hat{y}_{R_m} = R_m으로부터 예측된 반응값
> \\
> $$
> 이 Loss Function의 형태는 우리가 알고있는 Lasso의 것과 굉장히 비슷합니다.
>
> 
>
> 우리가 정한 hyperparmater {alpha}를 추가함으로써 subtree의 복잡도와 훈련데이터에 대한 에러를 동시에 고려하게 되는 메커니즘입니다.
>
> **지금 질문자가 헷갈린 부분은 위 식을 단지 한번만 계산해서 최적의 subtree를 찾아내야 한다고 생각한 것 같습니다.**
>
> 하지만 우리는 alpha가 정해졌을 때, T를 바꿔가면서 위 Loss Function을 수차례 계산하며 최적의 subtree T를 찾아내게 되는 것입니다.
>
> - 즉, T⊂T0 중 최적의 T를 찾는 과정 [T0는 subtree들을 포함하는 매우 큰 나무]
> - 이는 마치 Lasso Regression에서 LSE with L1Loss를 추정하는 과정
>
> alpha가 0일 때를 생각해보면, 무조건 원래의 큰 나무 T0일 때를 최적의 해로 생각하게 될 것입니다.
>
> 그리고 alpha를 바꿔감에 따라 (패널티 값을 바꿔줌에 따라) 해당 alpha값에서 최적의 subtree를 구해 최소 Loss값이 바뀌어갈 것이고,
>
> 이렇게 alpha를 바꿔가며 hyperparameter를 결정해야 합니다. (Cross-Validation을 사용하면 튜닝이 가능할 것입니다.)