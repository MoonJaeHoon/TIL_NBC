# 1. Transformer



이전에 배웠던 <u>RNN 기반 Seq2Seq with Attention 형태</u>에서 RNN구조를 아예 빼버리고 모든 부분을 Attention으로 채운 것이라고 생각하면 쉽습니다.



> 논문에 나와있는 Transformer의 구조는 다음과 같다.

<img src="Transformer(Attention is All you need).assets/image-20210218130648317.png" alt="image-20210218130648317" style="zoom:80%;" />



## 1.1 Introduction

Transformer에 들어가기 앞서, 앞선 RNN 기반의 Encoder Decoder는 단점이 존재하였다.



RNN을 이용한 Encoder는 `I` `go` `home` 각각의 단어에 잘 맞는 정보를 인코딩해놓았었음.

`home`에 `I`와 `go`의 정보가 전달되는 구조인데, 이는 앞선 time-step에서의 정보가 유실되는 현상이 일어날 수 있음 (Lorg term Dependecy, Gradinet Vanishing, Exploding 등)

![image-20210218135422231](Transformer(Attention is All you need).assets/image-20210218135422231.png)



위의 현상(긴 문장일수록 앞선 시점의 정보가 손실되는 현상)은 아래와 같이 Forward, Backward 두가지 방향을 이용하여 Encoding하는 방법을 생각해보게 한다.

- `go`라는 단어를 encoding 하는 과정을 생각해보자
- 왼쪽의 Forward에서는 `I`와 `go` 두 단어를 이용하여 hidden state 값, Backward는 `home`과 `go` 두단어를 이용하여 hidden state값을 구하게 된다.
- 이 둘을 concat하여 두배의 차원에 해당하는 벡터를 `go`가 가지게 된다.

![image-20210218135641921](Transformer(Attention is All you need).assets/image-20210218135641921.png)



## 1.2 Attention is All you need



### 1.2.1 Attention의 구조

RNN을 아예 사용하지 않고 Attention 만으로 구축하는 Transformer의 구조를 살펴봅시다.

- 앞서 배웠던 Attention 에서처럼 Attention Score , Attention Distribution 등을 구하는 과정이 보입니다.
- 여기서도 input과 output(hidden state) 벡터의 차원이 같음을 볼 수 있다.

![image-20210218142023104](Transformer(Attention is All you need).assets/image-20210218142023104.png)



```
앞서 배웠던 Seq2Seq의 Attention 메커니즘을 생각해보자.

어느 단어에 집중해야할지 Context Vector(Attention Score)를 구하는 과정에서 현재 time-step의 Decoder hidden state h_t(d)를 사용했었다.
Transformer에서는 이를 현재 time-step의 input 값 x_t로 대체하여 사용합니다.

아래의 그림에서 예를 들어서 1번째 time-step에서는 무엇에 집중해야할지 구하는 Attention 메커니즘을 수행한다면, Decoder 부분에서의 첫번째 hidden state 였던 h1(d)가 해야할 역할을 Input의 첫번째인 I가 대신하고 있다.
```



![image-20210218142556373](Transformer(Attention is All you need).assets/image-20210218142556373.png)



>  ※ 유의할 점 : 일반적으로 사용되던 Attention 메커니즘을 사용했다면, 해당 time-step에 해당하는 input x1 그자체를 이용하여 구할 때 Dot product같은 내적연산을 통해 유사도를 측정하고  해당 time-step에서 어디에 집중해야 할지 정하게 되는 가중평균을 도출했을 것이다.
>
> - 이런 방식을 따르게 되면 당연히 1번쨰 time-step에서는 `x1과 x1`, `x1과 x2`, `x1과 x3` 유사도를 내적을 통해 보게 될 것이고, 결국 당연히 x1과 x1의 유사도가 가장 큰 값을 가질 수밖에 없게 되는 양상을 띌 것이다.
> - 이러한 문제를 어떻게 해결가능한가?
>   - Query, Key, Value Vector를 각각 정의하여 업데이트해줌으로써 해결하였다.



```
이처럼 Decoder hidden state h_t(d)를 사용하지도 않고, input의 해당 time-step이 그 역할을 하고 있고, 이것으로 Attention Score, Distribution, Context Vector를 Encoding된 Vector로서 구해내는 과정이므로 Self-Attention Module이라고 부릅니다.

```



<img src="Transformer(Attention is All you need).assets/image-20210218152330731.png" alt="image-20210218152330731" style="zoom:67%;" />

### 1.2.2 Query, Key, Value

>  Query Vector : 주어진 벡터들 중 어느 벡터를 집중하여(선별적으로) 가져올지 나타내는 역할(유사도를 구하는 벡터로서의 역할)

- 해당 time-step에서의 Encoder input  `x_t(e)`으로부터 생성된 hidden state `q_t(e)`가 바로 query vector라고 생각해볼 수 있다.
- 마치 앞서 배웠던 RNN 기반 Attention에서 사용하던 Deocder의 해당 time-step hidden state `h_t(d)`를 대체할 수 있어보인다.
- `h_t(d)` 역할을 하게 된다는 것은 결국 이를 이용해 `x1`, `x2`, `x3` 각각과 내적연산(유사도)을 통해 그 중요도를 판단하게 된다는 것이다.
- 하지만 여기서도 역시 `x1`, `x2`, `x3` 를 그대로 사용하지 않고 어떠한 hidden state 값으로 바꾸고 내적연산이 이루어지는 것이 바람직해보인다. => Key Vector



> Key Vector : Query Vector와의 내적연산을 통한 유사도를 구함으로써 어느 단어에 집중해야 할지 정하기 위해 업데이트되는 hidden state Vector 부분이다.

- Attention Score를 구하기 위해 업데이트 되는 벡터



> Value Vector : Attention Score를 구하고, Attention Dist를 구했다면 얘를 이용해서 x1, x2, x3의 가중평균을 내고 Context Vector를 구해야한다. 
>
> 하지만, 역시 여기서도 업데이트될 수 있는 파라미터가 포함되게끔 Value Vector라는 hidden state 부분으로 만들어주었다.



※ 결국, 내가 생각하기에 중요한 포인트는 바로 query, key, value 각각의 Vector 들이 서로 값이 같지 않게끔 다른 기능을 할 수 있게끔, **업데이트될 수 있는 파라미터가 포함되게끔 hidden state로 변환**하는 생각을 해냈다는 것이다.





```
다음 그림에서 보이는 각각의 W matrix는 (W^Q, W^K, W^V) Query, Key, Value Vector로 선형변환을 해주는 매트릭스이다. 이로부터 각기 다른 기능을 하는 vector들을  `x1`, `x2`, `x3`로부터 생성해주는 것이다.
```



<img src="Transformer(Attention is All you need).assets/image-20210218160529062.png" alt="image-20210218160529062" style="zoom: 67%;" />

​	1) 결국 이런식으로 각기 다른 역할을 하게 선형변환해준 Query와 Key Vector를 이용하여 Attention Score(유사도)를 구해냅니다.

​	2) 이것에 softmax를 취해 Attention Distribution을 구한다.

​	3) Attention Distribution 을 Value Vector의 가중평균을 구하는데 사용하여 Context Vector를 만들어냅니다.

​	4) 위 과정을 각 time-step에 대하여 수행하면서 `h1`, `h2`, `h3`라는 Context Vector들을 구할 수 있게 된다.



그런데 여기서, key vector와 value vector를 만들어주는 선형결합 행렬 W^K , W^V matrix는 한 번만 업데이트 될텐데 각기 다른 `h1`, `h2`, `h3`라는 Context Vector들을 적절히 구해낼 수 있을지 의문이 들 수도 있을 것이라 생각합니다.

- 이러한 의문은 바로 Query Vector가 해결해줄 수 있가 있습니다.
- 첫번째 time-step에서는 Score(유사도)와 Distribution(가중치)를 구함에 있어서 `I`에 대한 Query Vector를 이용(즉, 어떤 W^Q를 사용), 다른 time-step에서는 또다른 Query Vector를 사용(즉, 어떤 또다른 W^Q를 사용)함으로써 다른 유사도를 뽑아낼 수 있게 되는 것이다.



> 위와 같이 가중치 매트릭스 (W^Q, W^K, W^V 매트릭스)를 통해 각각의 벡터들에 연산이 되는 것을 다음과 같은 행렬연산으로 생각해볼 수가 있습니다.
>
> - 결국 RNN 기반 모델의 문제가 되었던 Long-time Dependancy에 상관없이 집중해야할 정보를 무리없이 선택할 수 있습니다.

![image-20210219123410712](Transformer(Attention is All you need).assets/image-20210219123410712.png)



### 1.2.3 실제 사용되는 Matrix 형태를 이해해보기

결국 ContextVector는 앞서 구해진 Score, Distribution을 통해 Value Vector를 가중평균을 낸 것이라고 볼 수 있음.

- Query와 Key Vector는 내적연산이 이루어져야 하므로 Dimension이 같아야 한다.

- Value vector는 별개의 Dimension을 가질 수 있다.

![image-20210219124452572](Transformer(Attention is All you need).assets/image-20210219124452572.png)





위에서는 query Vector로서 살펴보았고, 실제 연산에서는 query vector들이 행으로서 concat되어 만들어진 아래와 같이 Query Matrix를 생각하게 됩니다.



![image-20210219142750937](Transformer(Attention is All you need).assets/image-20210219142750937.png)



> ※ 유의할 점
>
> 그런데 위 그림에서 이상한 점 한가지, Query와 Key의 Dimension이 다르다? (Q_d=3, K_d=4)
>
> 이게 가능한건가?
>
> - 정답은 바로 decoder 안에 있는 encoder-decoder 간 attention을 하기 때문에 이것이 가능한 경우입니다.
> - 해당 Attention에서는 Encoder에 들어가는 src sentence의 길이와 Decoder에 들어가는 trg sentence의 길이가 다를 경우가 있을 수 있기 때문입니다.
>
> 결국, Decoder를 수행함에 있어서는 Q랑 K의 차원수가 같아야 하지만 Input과 Output의 Sequence길이가 다를 수 있기 때문에 개수는 다를 수 있다.(이게 정말 헷갈리지만 중요한 부분이다. 번역 TASK를 생각해보자.)
>
> 
>
> 여기서 하나더 유의깊게 보아야할 부분은 k와 v의 개수(seq_len)가 무조건 같아야한다는 것이다. (차원의 수는 다를 수도 있음)
>
> k와 v의 개수가 같아야 한다는 것은 각 token의 개수인 seq_len이 동일해야 하는 것을 뜻합니다. (ex. Decoder에 투입되는 seq가 (=>번역의 결과로 나와야하는 seq가) I go home이라면 seq_len=3개)
>
> - 위 그림을 다시 한번 보면, Q의 길이(decoder input의 token 개수)와 K의 길이(encoder input의 token 개수)가 다른 경우이지만 K와 V의 길이는 동일한 것을 볼 수 있습니다.
> - 하지만 Q의 차원수, K의 차원수는 일치하고 있습니다.





![image-20210219143759628](Transformer(Attention is All you need).assets/image-20210219143759628.png)

```
결국 위의 그림에서 Attention의 결과를 보면 행은 Q, 열은 d_v의 차원을 가지고 있습니다.
해당 Attention 결과 Matrix는 각 행마다의(q1,q2,q3로부터 나온) Context Vector를 가지고 있는 형태인 것입니다.
문장seq의 길이이자 단어 갯수인 seq_len 개의 Context Vector가 있으며 각 하나의 Context Vector는 Value Vector의 차원을 따르게 되는 것입니다.

```

- Row-Wise Softmax라 함은 Row마다 Softmax 연산을 적용해준다는 것이다.
- 이렇게 함으로써 하나의 query vector 내에서 k의 개수(k1, k2, k3, k4이므로 여기선 4)만큼의 가중치를 구할 수 있게 된다.
  - 아래 예시를 보면 이해가 될 것이다. (0.2+0.1+0.4+0.3=1)

![image-20210219143919966](Transformer(Attention is All you need).assets/image-20210219143919966.png)





위 결과로부터 Value Vector로 이루어진 Matrix까지의 연산을 수행하면서 마지막 결과를 이해해보자.

- 다음과 같이 결국 결과 Matrix는 각각의 Query Vector에 대한 Output Vector를 계산하여 저장하고 있는 형태라는 것을 알 수가 있습니다.
- Matrix 형태로 연산을 하는 이유는 GPU와 같은 장치를 통한 병렬처리가 가능하다는 장점을 가지게 해주는 것입니다.

![image-20210219145215404](Transformer(Attention is All you need).assets/image-20210219145215404.png)



### 1.2.4 내적 연산 중 Problem과 Solution

Q와 K의 내적연산을 한 결과는 임의의 조정이 필요한 결과값이 나오게 됩니다.

- 예를 들어 다음과 같이 query, key의 원소들이 각각 평균이 0 분산이 1인 확률분포를 따르는 확률변수라고 생각해봅시다.
- query와 key 벡터가 2차원일 경우 내적연산의 결과로 생성된 ax+by라는 확률변수의 분산은 2라는 상대적으로 매우 작은 값이 나오게 됩니다. (경우1)
- 그리고 만약 query와 key 벡터의 차원이 100이라면 다음과 같이 분산=100(표준편차=10)이 나오고 차원이 커질수록 분산이 매우 큰 값을 가지게 되는 문제가 생깁니다. (경우2)

<img src="Transformer(Attention is All you need).assets/image-20210219151244246.png" alt="image-20210219151244246" style="zoom:50%;" />



```
예를 들어,
경우(1) - 내적연산을 통해 나온 Score값이 (1.1, -0.8, -1.7)과 같이 분산이 매우 작게 나옴. 
경우(2) - 내적연산을 통해 나온 Score값이 (9, -11, 7)과 같이 분산이 매우 크게 나올 수 있는 경우를 생각해보았을 때

※ 두 가지 모두 Distribution 을 구하기 위한 Softmax함수를 취해주었을 때, 문제점이 대두된다.

경우(1) : softmax를 취하게 되면 0~1 사이에 매우 고르게 분포되려는 경향이 생기게 됩니다.(실제 집중해야 하는 것에 집중하지 못하고 균등하게 가중치를 배분하는 경우)
경우(2) : softmax를 취하게 되면 0~1 사이에 고르지 않고 극단적으로 분포되려는 경향이 생기게 됩니다.(원래 고르게 분포되어야 하더라도 소수의 element에만 극단적으로 높은 값을 가져 집중하게 되는 경우)

```



- 위에서 말하고 싶은 것은 결국 의도치 않게 query와 key으로부터 나온 Attention Score 값의 분산에 따라 Softmax값이 변동될 가능성이 있다는 것입니다.

- 따라서 다음과 같이 차원수에 루트를 씌운값으로 나눠주어 이를 해결합니다.
  - 위의 (a,b)와 (x,y)에 대한 예시에서는 루트(2)로 나누어주게 됨. (분산은 2로 나누어준 값이 될 것이다)
  - 또한 100차원의 예시에 대해서는 루트(100)으로 나누어줍니다. (분산은 10으로 나누어준 값이 될 것입니다.)

![image-20210219145810609](Transformer(Attention is All you need).assets/image-20210219145810609.png)