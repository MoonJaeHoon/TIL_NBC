# Semantic Segmentaion



## 1.1 Semantic Segmentaion이란?

Semantic Segmentaion은 영상단위로 어떤 분류를 하는 것이 아니라 픽셀단위로 어떤 카테고리인지 분류를 한다.



<img src="Lecture4_Semantic%20segmentation.assets/img1.png" alt="image-20210309211327174" style="zoom:67%;" />

- 주의할 점 : 자동차가 여러개 있더라도 모두 같은 카테고리(자동차), 사람이 여러명 있더라도 같은 카테고리(사람)이라고 분류를 하는 Task이다.
- 원래 Semantic Segmentation 목적 자체가 그렇게 설계되었음



> 참고: Instance Segmentaion은 인스턴스까지 구분을 해준다.



## 1.2 Semantic Segmentation 적용분야

<img src="Lecture4_Semantic%20segmentation.assets/img2.png" alt="image-20210309211505745" style="zoom:67%;" />



- 의료 이미지

- 자율 주행

- Computational photography
  - 어떤 물체를 바꾸거나 수정할 수가 있음 (포토샵 등을 이용해)
  - 다음과 같이 인물집중 사진으로서 주변환경 흐리게 하는 것도 그 예시
  - 배경이나 인물을 아예 다른 배경화면, 또는 인물로 바꾸거나 하는 것도 그 예시

<img src="Lecture4_Semantic%20segmentation.assets/img3.png" alt="image-20210309211737947" style="zoom:50%;" />



## 2. Semantic Segmentation Architectures

Semantic Segmentation Model은 여러 종류가 있다.



### 2.1 Fully Convolutional Network(FCN)

Semantic Segmentation를 위한 최초의 end-to-end 뉴럴넷이다 

>  여기서 짚고 가야할 부분
>
>  end-to-end : 우리가 원하는 Task를 단 하나의 아키텍쳐로 처음부터 끝까지 수행할 수 있는 방법을 말하는 것 같다.



- 이전에 보았던 Alexnet의 경우 결과에서 FC layer Part에서 Flattening Vector를 만들기 때문에 (벡터화시키기 때문에) 입력해상도가 포함되지 않으면 학습되어진 FC layer의 정보를 사용할 수가 없는 한계가 존재했다.

- 입력해상도가 바뀌는 순간 컨볼루션 레이어의 activation map의 dimension이 달라지게 되고 이를 flattening하면 길이가 아예 달라지게 되기 때문이다.



- 이렇게 학습시 사용했던 이미지와 입력 이미지의 resolution이 resolution이 달라서 문제가 생기는 것을 FCN이 해결하였다.

- FCN은 학습시 사용했던 이미지와 입력 이미지의 resolution이 달라도 문제없이 작동한다(호환성이 높다).

- 입력과 출력 페어만 있으면 신경망 내부가 자동으로 학습되는 구조인 것이다. (이전에는 내부 알고리즘을 직접 작성하고 결합했던 것에 반하여)



그럼 그림으로 살펴보자.

기존의 CNN은 마지막 부분에 FC layer를 몇 단 두었었는데, FCN은 FC 대신 Fully convolutional layer만 사용한 것을 볼 수가 있다.

이러한 방식이 어떤 차이가 있을까?

<img src="Lecture4_Semantic%20segmentation.assets/img4.png" alt="fully-connected-vs-fully-convolutional" style="zoom:67%;" />

- Fully **connected** layer :  어떤 고정된 vector가 input으로 주어지면, output도 fixed vector (flattened vector)로 처리된다. 즉, 공간 정보를 고려하지 않습니다.

- Fully **convolutional** layer : 입/출력이 모두 activation map(tensor)이다. 1x1 conv layer이다.



![fully-connected](Lecture4_Semantic%20segmentation.assets/img5.png)

- Fully connected layer는 각 채널들을 일직선으로 쭉 펴서(flatten) concat합니다.



<img src="Lecture4_Semantic%20segmentation.assets/img6.png" alt="fully-convolutional" style="zoom:67%;" />

- 이와 달리, Fully convolutional layer는 각 채널에서 같은 feature로 분류되는 vector(즉, 같은 위치의 벡터)들을 묶어 m개의 1x1 필터와 conv 연산(내적)을 수행하여, m개의 채널을 가지는 feature map을 구성합니다.
- 이 때, conv 연산이므로 sliding window 방식을 사용하고 이 때문에 좀 더 spatial한 데이터가 유지된다는 장점이 있습니다.

- 이 과정에서 Pooling 계층을 여러번 통과하고, stride가 있어 FC layer에 비해 좀 더 넓은 receptive field를 가지고 있기 때문에, high-resolution의 input이 들어오더라도 결과값은 어쩔 수 없이, 훨씬 **저해상도의**(**low-resolution**)의 예측 스코어(히트맵)를 가지게 된다.
  - 이런 저해상도 문제를 해결하기 위해 **upsamping**을 사용합니다.

> **+번외질문**
>
> 그렇다면 다음과 같은 생각이 문득 들 수도 있을 것이다. 저해상도 문제를 해결하기 위해 Pooling과 Stride를 제거해버리면 어떨까?
>
> - 만약 그렇게 한다면 receptive field 자체가 작아져서, 영상의 전체적인 context를 파악하지 못하게 될 것이다. 그건 의미가 없다.



#### 2.1.1 Upsamping

위의 저해상도 문제를 피하기 위해 사용하는 방법이다

<img src="Lecture4_Semantic%20segmentation.assets/img7.png" alt="upsampling" style="zoom:67%;" />



- 고해상도 이미지 input에 Conv 연산과 Pooling을 거치면서, 출력값은 자연스럽게 저해상도로 Downsampling된다. 이를 마지막에 고해상도로 키워주는 연산이 Upsampling이다. 방법은 여러가지가 있지만, 최근에는 대표적으로 2가지가 사용된다.

<img src="Lecture4_Semantic%20segmentation.assets/img8.png" alt="transposed-convolution" style="zoom:67%;" />

`Transposed convolution` : swapping the forward and backward passes of convolution

- input 픽셀 각각을 filter에 곱해주어 늘리는 것이다.

- 즉, 딥러닝 네트워크가 네트워크 목적에 맞도록 최적화하려면 Upsampling을 사용하는데 그 중 하나의 방법으로서 `Transposed Convolution`을 사용할 수 있는 것이다.

- 이 방법은 앞에서 설명한 interpolation과 다르게 학습할 수 있는 파라미터가 있는 방법입니다
- Transposed Convolution는 대부분 semantic segmentation에 사용됩니다.

> 참고 : https://gaussian37.github.io/dl-concept-transposed_convolution/



Transposed convolution에서 반드시 주의해야 할 점은, **checkboard artifact**를 조심해야한다는 것이다. 

- 이는 마치 체크무늬처럼 일정 간격으로 특정 격자의 색이 진하게 나오는 것을 의미한다.
- 이는 **sliding window 기법을 사용하는 CNN의 특성 상 stride와 kernel 크기를 잘 조절하지 않으면 overlap 되는 구간이 생기기 때문에 일어난다.**



위의 이미지에서도 볼수 있듯, [az + bx] 구간처럼 overlap되는 구간은 다른 구간들보다 상대적으로 출력값이 높을 수 밖에 없다. 픽셀값이 높다는 것은 곧 진한 색상을 의미하므로, 상대적으로 진한 격자가 규칙적으로 나타나게 되는 것이다. 이는 짙은 색상의 이미지일수록 더 심하다.

`Upsample and convolution` : Decompose into spatial upsampling and featrue convolutoin

<img src="Lecture4_Semantic%20segmentation.assets/img9.png" alt="upsample-and-convolution" style="zoom:67%;" />



이러한 문제는 upsampling(정확히는 interpolution) 과정과 convolution 과정을 분리함으로써 쉽게 해결할 수 있다. Transposed convolution은 어설프게 overlap되는 구간들이 일부만 있었다면, upsampling을 통해 중첩 문제가 없이 골고루 영향을 받게 함으로써 전체적으로 평준화시켜줄 수 있다.

- 기존의 upsampling 연산 decomposition → **[spatial upsampling + feature convolution]**
  - **spacial upsampling** : {Nearest-neighbor (NN), Bilinear sampling}같은 interpolation. 학습가능한 파라미터는 전혀 없고 그냥 해상도를 키워주는 작업.
  - **convolution 연산** : spatial upsampling 직후 수행하며, 파라미터가 있으므로 학습이 되어 전체적으로 learnable upsampling을 구현할 수 있다.



#### [다시, FCN으로 돌아가서]

그런데 아무리 Upsampling을 했다고 하더라도 한 번 떨어진 resolution을 충분히 원래대로 끌어올리기는 쉽지 않다.

그저 연산 과정에서 잃어버린 정보들을 다시 살리는 일이기 때문이다.

결국, 목표대로 정확한 High-resolution의 output을 얻기 위해서는 다음 두가지를 모두 만족해야 한다.

- **Fine/Low-level/Detail/Local 등 미세한 각 부분의 디테일을 살리면서도**,
- **Coarse/Semantic/Holistic/Global, 전체적인 context를 볼 수 있는 넓은 시야를 가져야한다.**



**※ 이 두 가지를 모두 가지기위해서 기존의 방법들을 모두 합쳐서 사용한다.**

<img src="Lecture4_Semantic%20segmentation.assets/img10.png" alt="image-20210312032346925" style="zoom:67%;" />

1. 높은 layer의 activation map을 upsampling하여 해상도를 크게 끌어올린다.
2. 이에 맞추어 중간 layer의 activation map을 upsampling하여 가져오고, concat한다.

위에서는 FCN-8s가 가장 많은 layer들을 concat하는 형태가 된다. 이처럼 중간 layer들의 skip connection을 추가할 때 훨씬 더 명확한 이미지를 얻을 수 있다.



> FCN의 특징

- Faster : 직접 짠 컴포넌트(알고리즘)에 의존하지 않고 자동적으로 학습하는 end-to-end 구조이다.
- Accurate : feature 표현과 분류가 함께 최적화된다.



> Hypercolumn for object segmentation

FCN과 동일한 시기에 비슷한 내용의 연구도 있었는데, 이 경우 Fully convolutional network보다는 `Hypercolumn`을 강조했다.

기존의 CNN layer는 feature representation을 위해 마지막 FC layer의 출력을 사용했다. 그러나 이 방식은 한 픽셀에 모든 정보가 압축되어 있어 너무 coarse spatially했다.

이와 달리, Hypercolumn은 모든 CNN 유닛의 해당 픽셀 위치에 해당하는 값들을 stacked vector로 표현하는 방식이다.

- 초기 layer에서의 좀 더 미세한 국지적 정보들(fine localized information)을 추출할 수 있었다.
- 후기 layer에서는 더 전체적인 context를 보므로, coarse semantic information을 추출할 수 있었다.

다만 이 논문은 end-to-end에서 쓰이는게 아니라, bounding box를 구하는 sub component algorithm을 사용한 뒤에 적용하는 모델로 소개되었다.





### 2.2 U-Net

- FCN을 베이스로 만들어졌다.
- 낮은 층의 feature map과 높은 층의 feature map을 더 잘 결합하는 방식을 제시했다.FCN의 skip connection과 유사한 방식이다.
- 좀 더 정교한 segmentation이 가능해졌다.



#### 2.2.1 구조

<img src="Lecture4_Semantic%20segmentation.assets/img11.png" alt="u-net-architecture" style="zoom: 80%;" />

기존의 CNN 파트와 conv 연산을 적용하여 전체적인 feature map(holistic context)을 뽑아내는 downsampling 부분은 거의 같다. 여기서는 `Contracting path`라고 부른다.

- 3x3 conv를 사용한다.
- feature 채널의 숫자를 계속 doubling한다.

그러나, upsampling 파트는 차이가 있다. 여기에서는 `Expanding path`, `Decoding` 이라고 부른다.

- 한번에 upsampling 하는 대신, 채널 수를 반으로 줄여가며 점진적으로 upsampling한다(즉, Contracting path의 대응되는 layer와 채널 수를 동일하게 맞춘다.).
- 대칭되는 Contracting path의 layer에서 skip connection을 통해 대칭되는 feature map들을 가져와서 fusion(여기서는 concat)해준다.



>**※ 주의할 점**
>
>이 때 input 이미지와 feature 이미지의 크기는 짝수여야 한다.
>
>만약 홀수라면, Contracting/Expanding 파트에서 나머지 정보들이 유실된다는 점을 꼭 기억해야 한다.





### 2.3 DeepLab

Sementic segmentation의 한 획을 그은 모델로, 기존 모델에 비해 다음과 같은 차이점을 가진다.

1) **CRFs** 후처리의 존재

2) **Atrous convolution**



> Conditional Random Fields(CRFs)란?

- 픽셀과 픽셀 사이의 관계를 다 이어주고, regular한 pixel map을 그리드로 보는 것이다.
  - 기존의 sementic segmentation에서는 feedforward 구조이므로 피드백이 없어 굉장히 blurry한 output이 나오기 마련인데,
  - CRFs는 기존의 이미지에서 edge같은 경계선들을 활용하여 score map이 경계에 잘 들어맞도록 확산시켜주는 역할을 한다.
  - 이 때 물체의 background와 내부에서 동시에 확산하므로, 결과적으로 경계선이 물체 형태에 맞게 명확히 잡히게 된다.



> Atrous convolution

- 기존 convolution 필터와 달리, 필터의 수용영역 사이사이에 space를 넣어 spatial context를 캐치하는 방법이다.

- Dilation factor를 몇번 반복하는 것 만으로 파라미터 수는 늘리지 않으면서 receptive field는 exponential하게 키울 수 있다.



> Depthwise separable convolution

<img src="Lecture4_Semantic%20segmentation.assets/img12.png" alt="depthwise convolution" style="zoom:67%;" />



- 기존의 convolution 연산은 하나의 필터를 모든 input 채널에 대입시켰습니다.

- 그러나 *Depthwise separable convolution*은 이 과정을 다음의 둘로 분리하였습니다.
  1. Depthwise convolution : 채널별로 필터를 만들고, 이를 convolution하여 각 채널별로 따로 activation map을 만든다.
  2. Pointwise convolution : 뽑아낸 값들을 토대로 다시 1x1 conv를 사용함으로써 하나의 값으로 출력이 되도록 만들어준다.



**결론 : 위의 이미지를 보면 알 수 있듯, convolution의 표현력은 유지하면서 계산량은 훨씬 더 줄어들게 되었습니다.**



