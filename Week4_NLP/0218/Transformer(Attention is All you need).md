# 1. Transformer



이전에 배웠던 <u>RNN 기반 Seq2Seq with Attention 형태</u>에서 RNN구조를 아예 빼버리고 모든 부분을 Attention으로 채운 것이라고 생각하면 쉽습니다.



> 논문에 나와있는 Transformer의 구조는 다음과 같다.

<img src="Transformer(Attention is All you need).assets/image-20210218130648317.png" alt="image-20210218130648317" style="zoom:80%;" />



## 1.1 Introduction

Transformer에 들어가기 앞서, RNN 기반의 Encoder Decoder는 단점이 존재하였다.



RNN을 이용한 Encoder는 `I` `go` `home` 각각의 단어에 잘 맞는 정보를 인코딩해놓았었음.

`home`에 `I`와 `go`의 정보가 전달되는 구조인데, 이는 앞선 time-step에서의 정보가 유실되는 현상이 일어날 수 있음 (Lorg term Dependecy, Gradinet Vanishing, Exploding 등)

![image-20210218135422231](Transformer(Attention is All you need).assets/image-20210218135422231.png)



위의 현상(긴 문장일수록 앞선 시점의 정보가 손실되는 현상)은 아래와 같이 Forward, Backward 두가지 방향을 이용하여 Encoding하는 방법을 생각해보게 한다.

- `go`라는 단어를 encoding 하는 과정을 생각해보자
- 왼쪽의 Forward에서는 `I`와 `go` 두 단어를 이용하여 hidden state 값, Backward는 `home`과 `go` 두단어를 이용하여 hidden state값을 구하게 된다.
- 이 둘을 concat하여 두배의 차원에 해당하는 벡터를 `go`가 가지게 된다.

![image-20210218135641921](Transformer(Attention is All you need).assets/image-20210218135641921.png)



## 1.2 Attention is All you need

위와 같은 양방향 RNN 기반으로도 정보 손실의 문제가 보완만 될 뿐 완벽히 해결되지 못했다.

### 1.2.1 Attention의 구조

따라서, RNN을 아예 사용하지 않고 Attention 만으로 구축하는 Transformer의 구조를 살펴봅시다.

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
다음 그림에서 보이는 각각의 W matrix들(W^Q, W^K, W^V)은 Query, Key, Value Vector로 선형변환을 해주는 매트릭스이다. 각기 다른 기능(Q, K, V)을 하는 vector들을  `x1`, `x2`, `x3`로부터 생성해주는 것이다.
```



![image-20210315223306151](Transformer(Attention is All you need).assets/image-20210315223306151.png)

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

(사용되는 Matrix를 한눈에 나타낸 그림)

![image-20210315223559967](Transformer(Attention is All you need).assets/image-20210315223559967.png)


$$
|Q| = 쿼리의\ 갯수
$$

$$
d_k \ = \ 쿼리의\ 차원
$$

- *Q*와 *K*의 내적을 위해서 *K*는 transposed 되어서 계산되어지고 있다.
- 이 둘의 내적의 결과로 나오는 ∣*Q*∣×∣*K*∣ 행렬에서의 **i번째 row**는 **i번째 query에 대한** input 벡터들(key들)의 유사도를 나타내는 row가 된다.
  - 이 연산이 끝나면 softmax를 이용하여 가중치 벡터로 변환된다.
- 이렇게 나온 가중치 벡터와 *V* 벡터와의 내적을 통해 나온 ∣*Q*∣×*d**K* 출력 행렬은 다시 기존의 Q와 동일한 형태를 이루며, 출력 행렬의 i번째 row는 input Q의 i번째 row(query)에 대한 attention의 output이 된다.



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



<img src="Transformer(Attention is All you need).assets/image-20210219143759628.png" alt="image-20210219143759628"  />

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
- Matrix 형태로 연산을 하는 이유는 GPU와 같은 장치를 통한 병렬처리가 가능하다는 장점을 가지게 해주기 때문입니다.

![image-20210219145215404](Transformer(Attention is All you need).assets/image-20210219145215404.png)



### 1.2.4 내적 연산 중 Problem과 Solution

Q와 K의 내적연산을 한 결과는 임의의 조정이 필요한 결과값이 나오게 됩니다.

- 예를 들어 다음과 같이 query, key의 원소들이 각각 평균이 0 분산이 1인 확률분포를 따르는 확률변수라고 생각해봅시다.
- query와 key 벡터가 2차원일 경우 내적연산의 결과로 생성된 ax+by라는 확률변수의 분산은 2라는 상대적으로 매우 작은 값이 나오게 됩니다. (Case1)
- 그리고 만약 query와 key 벡터의 차원이 100이라면 다음과 같이 분산=100(표준편차=10)이 나오고 차원이 커질수록 분산이 매우 큰 값을 가지게 되는 문제가 생깁니다. (Case2)

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

![image-20210315223156909](Transformer(Attention is All you need).assets/image-20210315223156909.png)



![image-20210315231115242](Transformer(Attention is All you need).assets/image-20210315231115242.png)



## Multi-Head Attention

**`Multi-Head Attention`**은 기존의 Attention 모듈을 좀 더 유용하게 확장한 모듈이다.

<img src="Transformer(Attention is All you need).assets/image-20210315231233002.png" alt="image-20210315231233002" style="zoom:80%;" />

<img src="Transformer(Attention is All you need).assets/image-20210315231348345.png" alt="image-20210315231348345" style="zoom:67%;" />



> Multi-head attention은 **여러개(h개)의 attention 모듈을 동시에 사용**합니다.

먼저 위의 그림을 예시로 간단히 설명해보면, Multi-head Attetnion은 3개(일반, 반투명, 투명)의 Attention이 동시에 수행되고 있는 구조입니다.

이렇게 병렬적으로 수행되는 각 Attention_i들의 연산에서 Q, K, V를 돌려쓰면서 각 i마다의 Weight Matrix (W_i)를 다르게 업데이트합니다.

그러면 각 i 버전마다의 encoding Vector가 모두 다르게 나오게 될 것이고, 이들을 concat해서 최종적으로 해당 query에 대한 Output이 나오게 된다.

 

> 위에서의 간단히 설명한 과정들을 아래에서 조금 더 자세하게 뜯어서 보겠습니다.

아래 그림과 같이 8개의 Head 수만큼 Attention을 동시에 수행하게 되면 (Z0, ... Z7)이 나오게 됩니다.

![image-20210315231723460](Transformer(Attention is All you need).assets/image-20210315231723460.png)

이렇게 나온 (Z0, ... Z7)를 W^O로 linear transformation을 하여 하나의 Output Z를 만드는 형태로 진행된다.

![image-20210315231746208](Transformer(Attention is All you need).assets/image-20210315231746208.png)



여기에서 *W^O*는 여러번의 각 i번쨰의 attention 모듈에서 도출된 각기 다른 W_i Matrix들을 concat한 것이다.



> 이런 Multi-head attention을 사용해야 하는 이유는 무엇일까?

동일한 입력문 기준으로도 **필요에 따라 중점을 두어야 할 단어들이 다를 경우**가 있기 때문에 이를 위해 다양한 측면에서 정보를 뽑아야 할 필요가 있기 때문입니다.

- 예를 들어, I am going to eat dinner라는 문장이 있다고 하자.
- 어떤 때는 '내가' 먹었다는 사실에 주목해야 해서 'I'에 집중해야 할 수도 있고, 어떤 때는 '저녁'을 먹었다는 사실에 주목하기 위해 'dinner'에 집중해야 할 수도 있다.





### Attention의 연산량

- *n* : sequnce 길이
- *d* : query 와 key 차원
- *k* : Convolution 연산의 Kernel Size
- *r* : restricted self-attention의 이웃 사이즈

위와 같이 정의할 때, 기존 layer들의 연산량은 다음과 같다.

![image-20210315234748837](Transformer(Attention is All you need).assets/image-20210315234748837.png)



> **Complexity per Layer (총 연산량의 복잡도)**

Total Computational Complexity per Layer를 의미 

논문에서는 (연산양에 따른) 시간복잡도를 의미한 듯 한데, 공간복잡도로 해석해도 큰 무리는 없는 듯하다.(연산을 처리하는 디바이스가 1개라고 가정한듯 하다.)

- **Self-attention**

  - Q º K^T  = (*n*×*d*)×(*d*×*n*)

    ![image-20210316002012950](Transformer(Attention is All you need).assets/image-20210316002012950.png)

  - Q와 K를 내적하므로 계산되는 연산량은 *d*이고, 이를 모든 각 길이 *n*의 제곱만큼 계산해야 하므로

    => *O*(*n*^2⋅*d*)

- **Recurrent**

  - W_{hh} º h_{t-1}  = (d×*d*)×(*d*×*1*)

    ![image-20210316004322242](Transformer(Attention is All you need).assets/image-20210316004322242.png)

  - time step의 개수가 *n*이고, 매 time step마다 (*d*×*d*) 크기의 *W_{hh}*를 곱한다. 

    => *O*(*n*⋅*d*2)

  - 이 때 *W*_*{hh}*의 dimension *d*는 hidden state vector의 크기로(하이퍼파라미터) 직접 정해줄 수 있다.



> **Sequential Operations(연산 시간 소요량)**

해당 연산을 얼마의 시간내에 끝낼 수 있는가를 나타낸 것 (연산양 자체는 무한히 많은 GPU의 병렬연산으로 한번에 처리할 수 있음을 가정한다)

- **Self-attention**
  - 시퀀스의 길이 *n*이 길어질수록 지수배로 연산복잡도가 늘어난다.이는 모든 Q와 K의 내적값을 모두 저장하고 있어야하기 때문이다.
  - 따라서 **일반적인 Recurrent보다 훨씬 많은 메모리를 필요**로 하게 된다.
  - 하지만 GPU가 이런 형태의 행렬 연산 병렬화에 특화되어 있고, 따라서 충분히 많은 GPU를 가지고만 있기만 하다면 이를 병렬화하여 계산할 수 있으므로, 시간 복잡도는 *O*(1)로 볼 수가 있습니다.
- **Recurrent**
  - 이전 time step의 *h_{t-1}*이 제공되어야 그것을 input으로 다음 *h_t*를 계산할 수 있기 때문에, 불가피하게 시간 복잡도는 *O*(*n*)이 된다.

즉, Self Attention은 Rnn보다 많은 연산메모리를 필요로 하지만, 계산은 더 빨리 수행할 수 있습니다.



> **Maximum Path Length (두 단어 간의 경로 거리)**

Long-term dependency와 관련이 있는 지표입니다.

- **Self-attention**
  - 두 단어간의 유사도를 구할 때, 행렬 연산으로 바로 곱할 수 있으므로 O*(1)이다.*
  - 가장 처음에 있는 단어라 하더라도 동일한 key, value vector를 보기 때문에 정보를 직접적으로 가지고 올 수 있습니다.
- ***Recurrent***
  - 어떤 단어 a가 어느정도 떨어진 단어 b에 도달하기까지 recurrent cell을 하나씩 통과해야하기 때문에 *O*(*n*)이 된다.
  - 과거의 word 정보를 현재 step에 반영하려면 (만약 가장 처음에 있는 단어를 가져오려면) n번의 step을 지나야 합니다.



## Block-Based Model

### Encoder 구조

![image-20210316005601498](Transformer(Attention is All you need).assets/image-20210316005601498.png)



### **Add & Norm & Feed Forward**

위의 그림에서 처음에 시작해 Multi-Head Attention으로 가는 세개의 화살표는 각각 Q,K,V를 의미하며, 각 head마다 들어가게 되었었다.

그럼 그 연산 이후에 진행되는 **Add&Norm** 층은 어떤 역할일까?

![image-20210316010416692](Transformer(Attention is All you need).assets/image-20210316010416692.png)

> Add - **Residual Connection**

- CV 분야에서 깊은 레이어를 만들 때 graident vanishing을 해결하면서 더 깊은 층을 쌓도록 하는 효과적인 모델 구성요소이다.
- 주어진 input vector를 Multi-Head Attention의 encoding output에 그대로 더하여 새로운 output을 만들어서, 학습시에 Multi-Head Attention이 **입력 벡터 대비 정답 벡터와의 '차이나는 정보'만 학습**하도록 할 수 있다.
- 이 때, 주의할 점은 **Multi-Head Attention output과 input 벡터의 크기가 완전히 동일하도록 유지**해야 더하는 과정을 추가해줄 수 있다는 것입니다.



> **Norm(Normalization)**

일반적으로 신경망에서 사용되는 normalization은, (평균,분산)을 (0,1)로 만든 뒤, 원하는 평균과 분산을 주입할 수 있도록 하는 `선형변환(Affine Transformation)`으로 이루어진다.

- Batch Normalization

  - 먼저, 각 원소에 평균을 빼고, 표준편차로 나눈다. → (평균,분산)==(0,1)

  - Affine Transformation하여 원하는 평균과 분산으로 만든다.

  - 간단한 예시를 들어보겠습니다.

    1. batch size가 3, Dimension이 3인 input가 어떤 hidden layer에 입력으로 주어졌다고 가정하고

    2. 만약 hidden layer 첫번째 노드의 Output이 (3,5,-2)였다면, 우선 이것의 평균을 0, 표준편차를 1로 바꿔주는 과정을 거친다.

    3. 그 후, *y* = 2*x*+3과 같은 Affine Transformation을 수행합니다. (2와 3과 같은 값은 학습되어지는 파라미터입니다.)

    4. 만약 위와 같이 2와 3으로 최적의 파라미터가 결정되었다면, 이때의 평균과 분산은 (2,3) 일 것입니다.

       -  *y* = <font color='red'>2</font>*x*+<font color='blue'>3</font> → (평균,분산) = (<font color='blue'>3</font>,<font color='red'>2</font>)

       - (기울기)^2 = 분산, (y절편) = 평균

  -  이런 식으로 평균과 분산을 Optimization 과정에서 최적화해갑니다.



- Layer Normalization

  ![image-20210316014557758](Transformer(Attention is All you need).assets/image-20210316014557758.png)

  - Batch Norm과 방법은 똑같이 수행하지만, 여러 layer가 붙어있는 행렬을 대상으로, 한 layer마다 수행한다.
  - affine transformation은 각 layer의 동일한 node 기준으로 수행한다.(normalization이 column 단위였다면 affine transformation은 row 별)
  - Batch Norm과는 일부 차이점이 있지만, 큰 틀에서 **학습을 안정화**한다는 점은 동일하다.



Add&Norm 구간을 거치고 나온 output은 다시 Fully connected layer(Feed Forward)에 통과시켜 Word의 인코딩 벡터를 변환한다. 이후 다시 Add&Norm을 한번 더 수행하는 것까지를 끝으로 Transformer의 (self-attention 모듈을 포함한) **`Block Based Model`**이 완성된다.



### Positional Encoding

<img src="Transformer(Attention is All you need).assets/image-20210316013422451.png" alt="image-20210316013422451" style="zoom:150%;" />



RNN과 달리 self-attention 모듈 기반의 Block Based Model로 인코딩하는 경우, 순서를 고려하지 않기 때문에 input 단어들의 순서가 바뀌어도 output을 동일하게 출력한다.

예를 들자면,

​	(1) Input을 `"go home I"`로 줬을 때의 `go, home, I`에 해당하는 각각의 Output 벡터들과 

​	(2) Input을 `"I go home"`으로 했을 때 `I, go, home`에 해당하는 각각의 Output들이

​	같아진다는 것이다. 

이는 K와 Q간의 유사도를 구하고 V로 가중치를 구해 가중평균(이때 softmax를 통과한 값이므로 가중합 자체가 가중평균이다)을 도출하는 과정에서, sequence임을 고려하지 않기 때문이다.

**`Positional Encoding`**이 이를 해결하기위해 벡터 내의 특정 원소에 해당 word의 순서를 알아볼 수 있는 , **unique한 값을 추가하여 sequence 순서를 고려하게** 만들어준다.

- 이 때, unique한 값은 주기를 다르게 한 sin cos 함수를 활용한다. 주기함수는 입력값 x의 위치에 따라 출력값이 변하기 때문이다.

  ![image-20210316012632270](Transformer(Attention is All you need).assets/image-20210316012632270.png)

- 단, 하나의 주기함수만 사용하면 동일한 함수값을 가지는 구간이 생기므로, **서로 다른 여러 주기함수의 출력값들을 모두 합쳐서 사용**한다.

  - 예를 들어, 20번째 위치에 해당하는 positional Vector를 구하려고 한다면, 위의 왼쪽 그림에서 20번째 위치(x축=20)에 있는 dim 4,5,6,7에 해당하는 값들이 각각 [0.13,−0.89,0.92,0.27]라는 Vector가 나옵니다.
  - 바로 이렇게 나온 Unique Vector를 20번째 Input Vector에 더해줘서 계산하는 개념입니다.

- 사용하는 방법은 위의 오른쪽 그림에서 **`빨간색으로 표시해둔 0번째 Position에 해당하는 row 벡터`**가 구해지면 이를 0번쨰 position에 해당하는 원래의 Input Embedding Vector에 더해줍니다.

이렇게 특수한 값을 추가하여 인코딩하게 되면, input 단어의 순서가 바뀌었을 때 output 값도 다르게 출력할 수 있는, 순서를 구별할 수 있는 모델이 된다.



### Warm-Up Learning Rate Scheduler

![image-20210316010138178](Transformer(Attention is All you need).assets/image-20210316010138178.png)

기존의 모델에서 학습률(learning rate)는 하이퍼파라미터로, 학습 내내 고정되어있는 값이었다. 그러나 학습의 과정동안 효율적인 학습률은 계속 바뀌기 마련이므로, 이를 학습 과정 내에서 효과적으로 바꾸어 줄 수 있는 방식으로 **`Learning Rate Scheduler`**가 나오게 되었다.



### Transformer: Encoder Self-Attention Visualization

<img src="Transformer(Attention is All you need).assets/image-20210316010653779.png" alt="image-20210316010653779" style="zoom:80%;" />

위 그림은 making을 Query로 사용할 때 Attention이 어떻게 반영되는지 보여준다. 이 때 빨간색으로 테두리되어 있는 부분은 ***Head_1***의 Attention이 반영된 부분을 나타낸다. 즉, 각 Head의 Attention을 다르게 반영한다는 듯이다. 처음 5개 정도의 Head에는 more과 difficult에 Attention이 많이 되고 있다. 또한 자기자신(marketing)을 Attention하는 Head가 존재하기도 하고, 시기 정보(2009)를 Attention하는 Head가 존재하는 것도 확인된다.

<img src="Transformer(Attention is All you need).assets/image-20210316010723454.png" alt="image-20210316010723454" style="zoom:80%;" />

위 그림에서 its가 Query로 사용되면, its가 Law를 가리키고 있다고 보여주는 Attention Head와 its를 한정해주는 application에 Attention Head가 동작하는 것이 확인 된다.

위 그림들을 볼 수 있는 코드이다. (참고 : https://colab.research.google.com/github/tensorflow/tensor2tensor/blob/master/tensor2tensor/notebooks/hello_t2t.ipynb)



## 디코더 구조와 Masked

![image-20210316005627761](Transformer(Attention is All you need).assets/image-20210316005627761.png)

Outputs가 디코더의 입력으로 들어올 때, 기존의 ground truth 문장에서 앞쪽에는 `<SOS>` 토큰을 붙여 들어오므로, 한칸 밀린(shfited right) 형태로 들어오게 된다.

디코더에서 Attention 모듈을 포함한 한차례의 과정을 거친 후 다음 Multi-Head Attention으로 갈 때, 디코더의 hidden state vector를 입력 Q로 넘겨준다. 그런데, 나머지 K와 V 입력은 외부, 즉 인코더의 최종 출력으로부터 온다. 즉, 이 부분은 디코더의 hidden state vector를 기준, 즉 Q로 해서 인코더의 hidden state vector K, V 를 가중하여 가져오는, **인코더와 디코더간의 Attention 모듈**이 된다.

이 후 이미지에 나온 대로의 연산을 거치다가, 디코더의 최종 output 값이 Linear Layer와 Softmax를 거쳐 확률분포의 형태로 출력된다. 이 값은 Softmax-with-loss 손실함수를 통해 학습된다.



### Masked Self-Attention

Self-Attention 모델에서, 임의의 단어 a는 Q와 K의 내적을 통해 자신과 모든 단어들의 관계를 다 알수 있다. 이 때, 학습 당시에는 배치 프로세싱을 위해 a 뒤의 단어들까지 모두 고려하도록 학습이 진행되나, 사실 **실제 디코딩 상황을 고려한다면 a 뒤의 단어를 알아서는 안된다**. 이는 뒤의 단어를 추론해야 하는 상황에서 뒤에 어떤 단어가 있는지 미리 알고있는 일종의 cheating 상황이기 때문이다. 이러면 당연히 학습이 제대로 되지 않게 되어버릴 것이다.

![image-20210316011234087](Transformer(Attention is All you need).assets/image-20210316011234087.png)

디코더 과정의 이미지 중 **`Masked Self-attention`**이 이를 해결하기 위한 방법으로, 기존의 attention 모듈에서 Q, K 내적과 softmax를 통과한 값에서 현재 단어 a의 뒤에 있는 단어들을 key 값으로 계산된 셀들을 모두 삭제한다. `Mask` 라는 단어는 이처럼 **뒤쪽의 정보를 가린다(mask)**는 의미다.

<img src="Transformer(Attention is All you need).assets/image-20210316005732735.png" alt="image-20210316005732735" style="zoom: 67%;" />



위의 이미지는 [I go home → 나는 집에 간다] 라는 번역을 수행하는 사례인데, Q,K의 내적을 통해 얻은 정사각 행렬을 표현하고 있다. 이 때 주대각선 위쪽의 값들은 query보다 key가 뒤쪽의 단어들인 경우로, 이 셀들의 정보를 사용하여 학습을 하게 해선 안된다.



![image-20210316010926404](Transformer(Attention is All you need).assets/image-20210316010926404.png)



<img src="Transformer(Attention is All you need).assets/image-20210316010946690.png" alt="image-20210316010946690" style="zoom:80%;" />

따라서 이 셀들의 정보를 그대로 둔 채로 학습시키지 못하도록 해당 값들을 0으로 대체한다. 그 이후, 남은 주대각선 이하의 셀들만 가지고, row단위로 총합이 1이 되도록 normalize 한 정보를 최종 output으로 내보낸다.

