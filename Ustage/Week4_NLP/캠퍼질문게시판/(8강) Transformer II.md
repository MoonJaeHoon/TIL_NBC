## 1. 트랜스포머에 대해서 질문이 있습니다.

어려운 내용이다보니 질문이 좀 많지만 답변 부탁드립니다.

1. 트랜스포머를 학습 할 때는 무조건 teacher forcing이 될 수 밖에 없는건가요?

2. 트랜스포머는 테스트 시에는 디코더 부분에서 for문을 돌게 되나요?

3. multi-head attention을 한 후 feed forward를 하는데 이 feed forward의 역할이 궁금합니다.



> 1. 일반적으론 그렇습니다. Teacher forcing을 안 쓰고 RNN decoder를 썼을 때처럼 다 돌아도 물론 안 될 것은 없겠지만 multi-head attention의 이점인 모든 time step에 대한 병렬 계산의 이점이 감소되기 때문에 일반적으론 학습 과정에선 전부 input을 넣고 teacher forcing을 해준다고 보시면 됩니다.
>
>  
>
> 2. 1번과 연결되는 답변이지만 학습 시에는 teacher forcing을 쓰기 때문에 한꺼번에 모든 단어를 넣고 병렬 처리를 하지만 inference 시엔 transformer라고 해도 decoder는 한 단어씩 생성을 해야 하므로 for문을 돌면서 language modeling을 수행합니다.
>
>   
>
> 3. Feed-forward layer는 구조를 보시면 아시겠지만 그냥 linear transformation을 두 번 수행하는 layer로 그 자체로 사실 어떤 기능을 한다기 보단 그냥 일반적으로 저희가 딥 뉴럴 넷에 linear layer를 추가하여 표현력을 증대시켜주는, 그런 의도로 추가해주는 거라고 보셔도 무방합니다. 



## 2. self attention을 할 때 gradient 질문

self attention을 할 때 sqrt(d_k)로 나눠주지 않을 때 gradinent가 왜 작아지는 건가요?

이 질문은 어제 강의자료의 10페이지에 있습니다.

"Hence, its gradient gets smaller"이 부분이요. softmax로부터 역전파가 이루어질 때 softmax값이 커지면 gradient값이 작아진다는데 왜 그렇게 되는지 이해가 잘 안됩니다.



> 맥락으로 보아 다음과 같은 의미로 볼 수 있습니다. 저희가 multi-head attention을 수행하다보면 d_k 크기를 가진 벡터들을 계속 해서 내적하게 되는데요, 이러한 내적이 값을 곱하고 더하는 과정이다보니 차원수가 크면 이걸 거듭하는 과정에서 어떤 값은 계속 커지고 어떤 값은 계속 작아지는 상황이 발생합니다. 즉, 절대값이 극단적인 값으로 계속 수렴하는 현상이 발생하죠. 이 때, softmax를 취해버리면 결국 작은 값은 0에 수렴해버리고, 큰 값은 1에 수렴해버리기 때문에 여기서 "peaked"라는 표현이 들어간겁니다. 이렇게 softmax가 극단적으로 1과 0으로 나뉘어버리면 gradient를 구하는 과정에서 저흰 softmax 함수의 미분값 역시 0에 수렴해버립니다.
> $$
> y_i = \frac{e^{z_i}}{\sum_ke^{z_i}}y \ \ \ \
> ( z_i \text{는 벡터의 각 원소의 값})
> $$
> 소프트 맥스 함수가 위와 같이 정의될 때, 미분을 하면 아래와 같습니다.
> $$
> \frac{\partial y_i}{\partial z_i} = y_i(1-y_i)\ \ \ \ \ \text{if}\ i=j,\\ -y_iy_j\ \ \ \text{if}\ i\neq j
> $$
> 
>
> 위와 같이 어느 값이 1 또는 0이 되어 버리면 두 경우 모두 0이 되어버리는 것을 알 수 있습니다.