

# Visualizing CNN

CNN visualization은 CNN를 시각화하는 방법입니다.

Visualization은 CNN에서 Debugging tool 역할을 합니다.

일반 debugging으로는 CNN의 구조나 weight들이 복잡하게 되어있어서 무슨 의미인지 잘 모르기 때문에 이를 사용합니다. (BlackBox 모델)

 

#### ZFNet example

- Deconvolution을 이용해서 Visualization을 시도한 2014년 논문입니다.
- Low 계층에는 방향성이 있는, 선을 찾는 filter들, block을 찾는기본적인 영상처리 filter 모양을 가지고 있습니다.
- High 계층에는 의미가 있는 표현을 학습한다는 것을 알 수 있습니다.



![image-20210312082114598](Lecture6_CNN%20visualization.assets/img1.png)

 

#### Filter weight visualization

- AlexNet의 1st conv.layer의 filter은 11 x 11 size의 입력 channel입니다.
- 첫번째 conv.layer의 filter는 다양한 operation들이 학습되었다는 것을 알 수 있습니다.

- 또한, Filter visualization에다가 Convolution을 취한 Activation map에서는 영상의 각도에 해당하는 결과나 소소한 detail, 색깔로 경계선을 구분한 결과 등을 알 수 있습니다. 

<img src="Lecture6_CNN%20visualization.assets/img2.png" alt="image-20210312103432436" style="zoom:67%;" />

 

> [한계]
>
> 하지만 Mid layer나 High layer의 경우에는 Filter의 차원수가 높아서 직관적으로 Visualization을 하기 어려움이 있습니다.

 

#### Neural Network Visualization 종류

- Analysis of model behaviors : 모델 자체의 특성을 분석
- Model decision explanation : 하나의 입력 데이터에서부터 모델이 어떤 결론을 내었을 때 어디서 그러한 결론으로 도출되었는지 출력을 분석

![image-20210312082517258](Lecture6_CNN%20visualization.assets/img3.png)

 

- Analysis of model Behaviors
  - 왼쪽으로 갈수록 model을 이해하기 위한 노력 (모델 자체의 행동 특성을 분석)
- Model Decision explanation
  - 오른쪽으로 갈수록 data 결과를 분석하는데 초점을 맞췄습니다. (하나의 입력데이터로 부터 어떻게 이런 출력이 나오는지 분석)

 

### 2. Analysis of model Behaviors

방금 보았던 그림에서 왼쪽 부분, model behavior을 이해하기 위한 노력에 관하여 보도록 하겠습니다.

#### 2.1 Embedding feature analysis

##### 2.1.1 Nearest neighbors (NN) Search

 ![image-20210312082818569](Lecture6_CNN%20visualization.assets/img4.png)

- DB가 먼저 존재하고, 그 데이터 베이스 내에 분석을 위한 예제 데이터를 많이 준비해놓는다.
- Query Data가 들어오면 그 Query 영상과 이웃 영상을 찾기 위해 DB를 찾는다.
- 각 query와 db image를 Neural Network에 넣고, High Level Layer의 결과로 나온 Feature Map의 유사도를 비교한다.
  1. Semantically similar concepts are well clustered
  2. Pixel wise comparison이 아닌 의미론적인 비교를 통해 찾는 다는 것을 보여준다.



> 위 결과 해석예시)

- **빨간색 부분**을 보면, similar한 concept을 가진 사진들끼리 clustering되어 있다는 것을 알 수 있습니다.
- **파란색 부분**은 강아지 모양이 다른 위치에 있음에도 불구하고 같은 강아지의 사진이 골라진 것을 알 수 있습니다.
  - 이들은 피쳐가 정보를 잘 담고 있다는 뜻입니다.
  - 즉, 학습된 feature가 물체의 위치변화에 강인하게, 대신에 concept을 잘 학습했다는 것을 알 수 있습니다.



##### Nearest neighbors (NN) Search의 과정

> High Dimensional의 Feature Space로 Image를 옮겨서 유사도를 측정한다.

<img src="Lecture6_CNN%20visualization.assets/img5.png" alt="image-20210312100834634" style="zoom: 50%;" />

- Query image들을 Fully Connected Layer 이전단계까지 forward를 시키고 나온 결과물을 featrue space에 찍는다.

<img src="Lecture6_CNN%20visualization.assets/img6.png" alt="image-20210312100906186" style="zoom:50%;" />

- Query 뿐만 아니라 DB에 저장된 예시 Image Database에 있는 Image도 마찬가지로 작업을 진행한다.



<img src="Lecture6_CNN%20visualization.assets/img7.png" alt="image-20210312103347834" style="zoom:67%;" />

- Feature Vector끼리 유사도를 측정해서 Nearest Neighbors를 적용한다.



> 한계)
>
> - 만약 포즈와 위치가 다른 이미지들도 명확하게 neighbor로 분류한다면, 이는 모델이 위치변화에 강인하게, 컨셉을 제대로 학습했다는 의미이므로 모델의 Robustness를 입증할 수 있게 된다.
> - 하지만 그 정도로 잘 되기가 매우 어렵지...
>
> - DB에 수많은 사진들을 넣어놓고 특정 Query image를 날려 해당 image와 비슷한 feature space의 images를 찾아내는지 확인한다.
>   - 그냥 간단하게 가지고 있는 사진들 중에 모델 이용해서 검색하는 것임.
> - 결과를 눈으로 직접 보고, 의미론적으로(semantically) 비슷한 이미지들이 잘 클러스터링되어있는지 확인할 수가 있다.
> - 이 방식은 결국 눈으로 예제를 보고 판단해야 하므로 전체적인 조감도를 보기는 어렵다는 단점이 있습니다.



##### 2.1.2 Dimensionality reduction

- 앞서 살펴보았던 Backbone network를 활용해서 feature를 추출하게 되면 고차원 feature vector가 나오게 됩니다.

- 이러한 고차원 feature vector는 해석하기 어렵다는 단점이 있습니다.

- 따라서 차원 축소를 통해 눈으로 쉽게 확인 가능하게 만들어줍니다.

![image-20210312085931660](Lecture6_CNN%20visualization.assets/img8.png)



 

> t-distributed stochastic neighbor embedding (t-SNE)
>
> - 가장 대표적인 예시 (비선형 기반의 차원축소)
> - t-distribution 이용함

example)

![image-20210312103315164](Lecture6_CNN%20visualization.assets/img9.png)

- 위는 0-9까지의 class로 이루어진 MNIST들의 feature들을 2차원 상으로 추출한 것입니다.
- 비슷한 class들끼리 분포가 잘되어있는 것을 알 수 있습니다.
- 여기서 3(연두), 5(하늘), 8(보라) 는 형태가 비슷하기 때문에 cluster가 비슷한 곳에 분포되어 있는 것을 알 수 있습니다.

 

#### 2.2 Activation investigation

Middle&High level를 해석하는 해석방법이다.



##### 2.2.1  Layer activation

※ Behaviors of mid- to high-level hidden units

각 Layer에서 Channel들의 Hidden node가 입력되는 이미지들에서 어떤 부분을 공통적으로 Activate하는지를 판단해서 역할을 시각화해줄 수 있습니다.



<img src="Lecture6_CNN%20visualization.assets/img10.png" alt="image-20210312103731957" style="zoom:67%;" />

 

- AlexNet의 conv5 unit 138(channel)을 거쳐서 나온 Feature Map을 적당한 값으로 Threshold를 주고 영상에 Overlay를 시키면, 위와 같이 입력되는 여러 이미지에서 공통적으로 얼굴을 Activation한다는 것을 알 수 있다.
- 마찬가지로 conv5 unit 53(channel)을 거쳐서 나온 것은 계단의 경로 부분을 Activation 해준다는 것을 알 수 있다.
- 이를 통해서 Hidden Layer의 Channel이 어떤 역할을 하는지를 시각화할 수 있다.



##### 2.2.2 Maximally activating patches

<img src="Lecture6_CNN%20visualization.assets/img11.png" alt="image-20210312104050117" style="zoom:67%;" />

역할을 파악하고 싶은 특정 Channel을 고르고 이미지의 랜덤 부분을 입력으로 줌,  높은 activation 나온 순으로 보여줌

각 Channel(Hidden Node)에서 **가장 높은 값을 가지는** 위치 근방의 Patch를 image별로 뜯어서 나열해보면 어떤 역할을 하는지 파악할 수 있다. (Mid Layer의 국부적인 특징을 분석할 때 유용)

- 위 그림을 보면, row별로 다른 hidden node들이다.
  - 맨 위의 채널은 강아지의 눈을 찾는 역할
  - 두번째 채널은 약간 커브가 들어간 부분을 찾는 역할



> 구현과정)
>
> - 역할을 파악하고 싶은 특정 Channel을 고릅니다. (conv5의 14 채널을 골랐음)
>
>   <img src="Lecture6_CNN%20visualization.assets/img12.png" alt="image-20210312104654564" style="zoom:50%;" />
>
> - BackBone Network에 예제데이터를 넣고 각 layer의 Activation map을 다 뽑음. 
>   
>   - Mid에 있는 우리가 워하는 14 채널의 정보를 가지고 오기 위해서는 Mid Channel에 있는 activation map을 추출해야 합니다.
>
> <img src="Lecture6_CNN%20visualization.assets/img13.png" alt="image-20210312104958039" style="zoom:50%;" />
>
> 
>
> - 이후, 추출된 Activation map에서 Maximum Value를 도출하게 한 그 위치에 거슬러 올라가서 입력 도메인의 receptive field 부분을 가지고 올 수 있을 것입니다.
>
> - 그 receptive field에 대한 입력 영상의 해당 패치를 아래와 같이 가져올 수 있습니다.
>
>   - 아래와 같이 각 hidden node 별로 나열을 해주었습니다.
>
> - 결과적으로 이 hidden layer가 어떤 부분을 주의깊게 살펴보는지를 알 수 있습니다.
>
>   - 이후는 위에서 해석했던 대로.
>
>   <img src="Lecture6_CNN%20visualization.assets/img14.png" alt="image-20210312105324726" style="zoom:67%;" />
>
> 





 

##### Class Visulization

예제 데이터를 사용하지 않고 네트워크가 내재하고 있는(기억하고 있는) 이미지가 무엇인지 분석해본다.

- 계속해서 예제데이터를 가지고 처리해왔는데, 이번엔 data 없이 모델에 저장된 Class 자체에서 visualization하는 방법입니다.
- 주어진 클래스에 제일 비슷하게 나오는 이미지를 학습시키는 것(모델은 고정시키고)

 <img src="Lecture6_CNN%20visualization.assets/img15.png" alt="image-20210312114036366" style="zoom: 80%;" />

위 그림을 통해 Network가 상상하고 있는 이미지를 분석해보면

- 강아지를 찾을 때는 강아지 뿐만 아니라 주변에 사람이 있는 경우도 찾고 있다.
  - (이런 경우에는 주어진 학습데이터의 이미지가 조금 편향되어 있을 수 있다는 것 또한 판별할 수 있다.)
- 새를 찾을 때는 새 뿐만 아니라 주변에 나무나 나뭇잎이 있는 경우도 찾고 있다.



##### Loss function 구축

여기서는 Gradient Ascent 개념을 사용하는데 합성 영상을 만들기 위해서 Loss를 만들어줍니다.

일단, 두 개의 Loss를 합성해서 사용합니다.

- *`I`* : 입력 영상
- *`f`* : CNN 모델
- *`f`_{dog}*( *`I`* ) : dog 관련 score만 출력하는 부분
  - (→ f(I)는 어떤 이미지가 CNN을 거쳤을 때 출력되는 Class Score를 의미한다. )
  - 결론적으로는 이 score를 max로 만드는 I를 찾는 것
- *`argmax f_{dog}(I)`* : dog 관련 score가 최대가 되는 I의 값
- *`Reg(I)`* : Regularzation(정규화) 부분

 

<img src="Lecture6_CNN visualization.assets/image-20210312102837400.png" alt="image-20210312102837400" style="zoom:50%;" />



`Reg(I)`는 다음과 같이 L2norm을 활용합니다.

제곱이 붙었기 때문에 argmax를 구할 때 `I`의 크기가 0이 될수록 유리해집니다. (그러면 argmax값이 커짐)

 

<img src="Lecture6_CNN visualization.assets/image-20210312102907275.png" alt="image-20210312102907275" style="zoom:50%;" />



원래 Loss는 최소값(argmin)을 구하지만, 여기서는 argmax를 구하기 때문에 gradient ascent라는 개념이 사용되어야 할 것입니다. (사실은 마이너스 붙여서 사용하면 뉴럴넷 안에서 경사하강법 적용가능함)

 

##### Loss를 이용한 역전파 업데이트 과정

>  Process of Gradient ascent

(1) **빈 image** 또는 **random값으로 초기화된 image**를 넣어서 최종 prediction score을 받아옵니다.

<img src="Lecture6_CNN%20visualization.assets/img16.png" alt="image-20210312115550555" style="zoom:50%;" />

 

(2) Backpropagation을 통해서 입력단의 gradient를 구합니다. 따라서 입력이 어떻게 변해야지 이 target class score가 높아지는지를 찾습니다. 즉, target score를 높이는 방향으로 빈 image를 update해줍니다.

- Tip) Loss를 측정할 때, <u>**Score값에 마이너스를 붙여서 내려가는 방향**</u>으로 만들어놓고 gradient를 계산하면 이전 neural network에서 학습했던 gradient descent 알고리즘을 그대로 사용할 수 있습니다.

<img src="Lecture6_CNN%20visualization.assets/img17.png" alt="image-20210312115611972" style="zoom:50%;" />

 

(3) image를 update해줍니다.

<img src="Lecture6_CNN%20visualization.assets/img18.png" alt="image-20210312115643675" style="zoom:50%;" />

 

(4) 이 update 과정을 가지고 (1)을 다시 반복합니다.

<img src="Lecture6_CNN%20visualization.assets/img19.png" alt="image-20210312115657068" style="zoom:50%;" />

 

(5) 이 과정을 여러번 반복하게 되면 target class에 들어있던 이미지의 형태를 구할 수 있습니다.

<img src="Lecture6_CNN%20visualization.assets/img20.png" alt="image-20210312115844049" style="zoom: 67%;" />

 

> Gradient descent는 현재 영상을 어떤식으로 update할지 local search하기 때문에 조금조금씩 변화를 주면서 이 score를 올리는 방법을 찾습니다. (최대한 가깝게 사진을 맞추겠죠)
>
> 이 말을 바꿔서 이야기하면 초기값을 어떻게 설정하냐에 따라서 다양한 결과를 얻어볼 수 있습니다.

역시 초기값 설정의 중요성..

 

### 3. Model decision explanation

지금까지는 모델 자체의 행동을 분석했다면

이번엔 Model이 어떤 특정 각도로 영상을 바라보고 있는지에 대해 해석하는 방법입니다.

####  3.1 Saliency test

##### 3.1.1 Occlusion map

영상이 주어졌을 때, 영상이 제대로 판정되기 위한 각 영역의 중요도를 추출하는 방법입니다.

- 아래 사진과 같이 코끼리 사진을 넣어줬을 때 A라는 큰 패치를 이용해서 Occlusion으로 가려줍니다. 
- 이 때 model이 이 사진을 코끼리라고 인식할 확률(CNN score)를 계산하여 패치가 어떤 위치를 가리고 있느냐에 따라 이 값이 바뀌는 정도를 기록해둡니다.

- 이 확률값들은 영상에서 Occlusion으로 어떤 위치를 가려주냐에따라 확률이 바뀌게 됩니다.
- 물체의 중요한 부분을 가리게 되면 Score가 많이 떨어지고, 물체와 상관없는 배경 부분을 가리게 되면 Score가 적게 떨어집니다.

- 이렇게 Occlusion 패치를 다양한 부분에 가려본 후, 패치의 위치에 따라 변하는 지도를 그릴 수 있습니다.

![image-20210312122017543](Lecture6_CNN%20visualization.assets/img21.png)

 

> 위 Heatmap에서 보이듯이 검정색 부분은 물체의 검출에 민감한 부분이고, 밝은 부분은 별로 상관없는 부분이라는 것을 알아낼 수 있었습니다.

 

#####  3.1.2 Saliency test via Backpropagation

gradient ascent를 이용하여 클래스 이미지를 생성했던 방법과 비슷한데, 이번엔 random image가 아니라 **특정 image**를 입력해 **해당 image를 classification하는 데에 큰 영향을 끼친 부분**을 heatmap으로 표시하는 기법이다.

 

- 다음 예시 사진에서는 흰색 부분이 물체 검출에 민감한 부분이고, Model이 흰색 부분을 보고 최종 score를 결정합니다. 

<img src="Lecture6_CNN%20visualization.assets/img22.png" alt="image-20210312122342406" style="zoom: 67%;" />

 

**※ via Backpropagation의 과정**

> 구하는 순서는 다음과 같습니다.
>
> 1) 먼저, 강아지 사진(우리가 원하는 도메인)을 넣게되면 해당 Class의 score가 나오게 됩니다.
>
> <img src="Lecture6_CNN%20visualization.assets/img23.png" alt="image-20210312123621269" style="zoom:50%;" />
>
> 
>
> 2) Score로부터 Backpropagation을 통해 입력 domain까지 계산해서 구합니다. 이렇게 얻어진 gradient에 절대값(또는 제곱)을 취해줍니다. 그리고 구한 값을 이미지 형태로 출력합니다.
>
> - 여기서 gradient에 제곱이나 절대값을 취해주는 이유는 무엇일까요?
>   - `+` 값이건 `-` 값이건 부호는 상관없이 **얼마의 크기만큼 변화를 줄 것인가** 만 고려하면 되기 때문이다.
>   - 결국 우리의 최종목적은 왼쪽에서 얼마나 변했는지 보고 이 부분이 민감하구나를 판단하는 것일 것이다.
>   - 따라서 왼쪽으로 역전파되어와서 +건 -건 상관없이 그 크기에 비례하게만큼 하얗게 변화를 주어야 한다는 말이다.
> - 따라서 부호보다는 크기가 더 중요해서 gradient에 제곱이나 절대값을 취해줍니다
>
> ![img](Lecture6_CNN%20visualization.assets/img24.png)
>
> 
>
> 3) 다시 (1)부터 과정을 반복수행합니다.
>
> 

 

##### 3.1.3 Rectified unit (backward pass)



> 기존 Backpropagation방법의 문제점
>
> **<u>Saliency Map</u>**
>
> - 원래 일반적으로 CNN의 Forward에서는 Activation function으로 **ReLU**가 많이 사용된다
>   - Relu 함수는 입력이 음수이면 0으로 마스킹 하는 효과를 가지고 있다. 
> - Relu를 쓰면 이렇게 음수가 0으로 마스킹되는 효과를 얻는데, 여기서의 문제는 Backpropagation을 할 때에도 마스킹하는 기준이 forward에서의 0이하 unit들을 기억했다가 마스킹해버린다는 것이다.
>   - 이런 masking pattern을 저장해놓고 이를 기준으로 판단하기 때문에
>   - 예를 들어, 만약 Backpropagation을 수행할 때 양수와 음수가 합쳐진 gradient가 오게 되면, 음수 mask로 저장되어있던 masking pattern으로 masking을 해줍니다
> - 이것을 Saliency Map이라고 한다.
> - (아래 그림에서 오른쪽 Backward pass: backpropagation) 에서 빨간색이 masking pattern이며, backpropagation이 왔을 때 빨간색 모양대로 '0'처리
>
> 
>
> **<u>Deconvolution</u>** 
>
> - Relu를 쓴다는 것은 위 방법이랑 비슷하지만 이건 backward pass에서를 기준으로 한다.
>
> - Foward 시의 ReLU pattern을 사용하는 것이 아니라 Deconvolution된 activation이 backward되어서 내려올 때의 값을 판별함
> - (아래 그림에서 Backward pass : "deconvnet") 양수와 음수 값을 따져서 activation의 음수값을 masking을 합니다.
>
> 
>
> ※ Saliency Map vs Deconvolution
>
> <img src="Lecture6_CNN%20visualization.assets/img25.png" alt="image-20210312174650809" style="zoom:67%;" />
>
> 
>
> 
>
> 수식으로 보자면 다음과 같이 나타낼 수 있을 것이다.
>
> <img src="Lecture6_CNN%20visualization.assets/img26.png" alt="image-20210312174920941" style="zoom:50%;" />
>
> 
>
> 차례대로 살펴보자면,
>
> 1. 첫번째 수식은 Relu가 적용된 모습
> 2. 두번째 수식은 기억해놓았던 마스킹 패턴인, Relu 부분 ( `h^l` > 0 )을 그대로 곱해서 사용했습니다 (Saliency Map)
> 3. 세번째 수식에서는 상위의 Activation의 Relu를 적용한 ( `h^{l+1}` > 0 )을 곱해서 사용했습니다.



+질문)

그렇다면 이 둘을 같이 합쳐서 사용하면 되지 않을까요?

- Saliency Map 와 Deconvolution 를 둘다 사용하는 **Guided BackPropagation** (하나라도 음수면 0으로 마스킹 처리)





##### 3.1.4 Guided BackPropagation

- 위에서 보였던 두 개의 Backward pass 과정을 합쳐서 사용하면 다음과 같습니다.

- 이 과정을 **Guided backpropagation**이라고 합니다.

 <img src="Lecture6_CNN%20visualization.assets/img27.png" alt="image-20210312175522813" style="zoom:80%;" />

 

> 다음 그림에서는 **Guided backpropagation**을 사용하면 Backpropagation이나 Deconvolution을 수행한 것보다 훨씬 직관적으로 잘 나오는 것을 보여주고 있습니다.

 ![image-20210312175704319](Lecture6_CNN%20visualization.assets/img28.png)



- 두 mask를 모두 사용한 것이 **(1)** **Foward를 할 때도 결과에 긍정적인 영향을 미친 양수들을 참조**하고, **(2)** **Backward를 할 때도 gradient를 통해서 더 강화하는 방향으로 움직이는 Activation을 고른 것**입니다.

- 그래서 위 두개의 조건들을 모두 만족하는 Activation들이 Guided backpropagation에 나타나게 되고 결과적으로 더 직관적으로 보여줄 수 있는 것입니다.

 

####  3.2 Class activation mapping

##### CAM

CAM은 사용하게되면 어떤 부분을 참조해서 어떤 결과가 나왔는지 보기 좋은 heatmap 형태로 표현해줍니다.

![image-20210312180151035](Lecture6_CNN%20visualization.assets/img29.png)

- CAM 아키텍쳐는 CNN의 일부를 개조해서 만들어집니다.
- CNN의 conv파트를 최종적으로 통과하고 FC layer에 진입하기 전, 즉 CAM은 CNN에서 마지막으로 나온 Convolution Feature map을 FC layer를 바로 통과하지 않고 Global average pooling(GAP) layer를 통과하도록 바꿔주어야합니다.
- 마지막으로는 역시 FC layer를 하나 통과하여 Classification Task을 수행하는 구조입니다.

![image-20210312180550750](Lecture6_CNN%20visualization.assets/img30.png)

 

> 수식은 다음과 같습니다.

<img src="Lecture6_CNN%20visualization.assets/img31.png" alt="image-20210312180418316" style="zoom: 80%;" />

*`c`* : 하나의 Class

*`k`* : 마지막 Conv layer의 channel 수

*`S_c`* : 어떤 클래스 C에 대한 score

*`W_k^c`* : 마지막 FC layer에서의 클래스 C에 해당하는 weight들

*`F_k`* : 각 채널별로 Conv Feature Map을 Global Average Pooling한 value

- [GAP 개념] - 간단히 설명하면, 모든 픽셀 (x,y)에 대해서 conv feature map을 평균취한 것이다.

- 해당 feature map의 분포 정보를 대표값으로 가지고 있게 된다고 생각하자.

- ∑_{*x*,*y*}가 Global Average Pooling 연산을 나타내는 수식 부분이다.

- 

- 첫줄의 Score는

  - $$
    S_c = \sum_k w_k^c F_k\ \ \ \ :\ GAP로부터\ 나온\ value들과\ w의\ 선형결합으로\ 계산되는\ Score인\ 것임.
    $$



> 위의 연산들은 모두 선형 연산이므로 다음과 같이 순서를 바꾸어줄 수가 있을 겁니다.
>
> <img src="Lecture6_CNN%20visualization.assets/img32.png" alt="image-20210313054239063" style="zoom:67%;" />
>
> - 위의 <font color="red">**빨간색 파트를 CAM이라고 부릅니다.**</font> 
>
>   - 연산순서를 바꾸어준 결과로 이는 Global average Pooling을 적용하기 전이므로 아직 공간에 대한 정보가 남아있습니다.
>
> - 이것을 영상에서 처리해 Visualization하면 하단과 같은 히트맵처럼 나오게 되는 것입니다.
>
>   ![image-20210313054446561](Lecture6_CNN%20visualization.assets/img33.png)
>
> 
>
> ※ 중요한점 : Global Average Pooling을 하기 전이라서 보존하고 있는 (x,y)에 대한 공간상의 정보와 어느 채널이 중요한지 나타내주는 채널별 weight를 가지고 있기 때문에 가능한 것임.
>
> 
>
> - 나오는 히트맵이 다른 방법에 비해 부드럽고 압도적으로 좋은 성능을 보여주기 때문에 자주 이용한다.
> - 사실 이미지 분류 Task만 하려던 것인데 위와 같은 분류 직전의 heatmap을 이용하면 분석이 여러가지 가능하다.
>   - heatmap을 원본 이미지에 오버랩하는 방식으로 시각화해본다.
>   - 그 다음 threshold까지 히트맵에 적용해주면 바운딩 박스를 그려볼 수가 있다.
>   - 구했던 히트맵을 통해 간단히 object detection도 가능한 것이다.
>     - **그것도 위치에 대한 어떤 annotation 정보도 주지 않았는데 위치를 찾아주고 있는 셈이다.**
>
> <img src="Lecture6_CNN%20visualization.assets/img34.png" alt="image-20210313055448702" style="zoom: 67%;" />
>
> 
>
> - 또한 이미지 분류를 할 때, 위치에 대한 어떤 annotation 정보도 주지 않았는데도 위치까지 어느정도 찾아주기 때문에 bounding box를 만들어 object detection 등을 추가적으로 하는데에 사용되기도 한다.
>   - 정보를 안 줬는데도, 그저 영상인식만 했는데도 객체(위치) 인식을 해주는 제 3의 TASK까지 모델이 해줄 수 있게 되었다는 것이 혁신적.
> - 이처럼 object detection과 같은 비교적 정교한 task를 좀 더 rough한 영상인식 task로 학습하여 처리하는 방식을 **Weakly Supervised Learning** 이라고 부른다.





>  **사실 단순 CAM을 활용하려면 보통 GoogLeNet이나 ResNet 모델 아키텍쳐를 활용합니다. 왜 그럴까요?**
>
> - 다만 CAM이 적용이 가능한 제약으로는 마지막 layer 구성이 FC layer로 만들어져야한다는 것이 단점입니다.
>
>   - 예를 들어서 AlexNet을 보면, 네트워크 구조의 마지막 부분에서  Flattening을 적용하고 FC layer에 집어넣어주는 식입니다.
>
>     ![image-20210313062359061](Lecture6_CNN%20visualization.assets/img35.png)
>
> - 그렇다면 이 모델을 활용하여 CAM하려면, 우리는 아래와 같이 FC layer로 구조를 변경한 후에 재학습을 해야 사용이 가능하다는 것입니다.
>
>  ![image-20210313061351318](Lecture6_CNN%20visualization.assets/img36.png)
>
>   **※ 하지만, 이렇게 모델구조를 바꾸게 되면 파라미터도 바뀌는 등 성능이 떨어져버리게 되는 결과가 초래된다.**
>
> 
>
> - 즉, 구조를 수정하지 않고 CAM을 추출하는 것이 중요하게 될 것이고 이를 위해선 AlexNet 아키텍쳐보단 GoogleNet, ResNet을 쓰면 된다.
>- **CAM을 활용하고 싶다면, <u>GoogLeNet이나 ResNet</u>의 구조에서는 마지막에 원래부터 Global average pooling이 들어가있기 때문에 CAM을 추출하기에 유용한 구조입니다.**
> - 즉, 구조를 수정하지 않고 CAM을 추출하는 것이 가능합니다. ​
>
> <img src="Lecture6_CNN%20visualization.assets/img37.png" alt="image-20210313062747675"  />



---

#### Grad-CAM

- 위에서 보았듯 CAM은 최종 층의 구조를 바꿔야 해서 모든 아키텍쳐에 적용할 수는 없다는 제약사항이 있었습니다.

- 이에 구조를 변경하지 않고 기학습된 네트워크에서 CAM을 뽑을 수 있는 **`Grad-CAM`** 방식이 제안되었습니다.

- 기존 pretrained된 모델 아키텍쳐를 변경할 필요가 없기 때문에, 영상 인식 task에 한정될 필요가 없어졌다. (아래의 설명을 쭉 보면서 이해할 수 있습니다.)

- 오로지Backbone이 CNN이기만 하면 아래와 같이 사용하여 결과를 보일 수 있다.

  

  예시)

  <img src="Lecture6_CNN%20visualization.assets/img38.png" alt="image-20210313060042790" style="zoom: 80%;" />



> **Grad-CAM의 과정설명**
>
> 우리가 기존의 CAM 식에서 알아내야 하는 부분은 *`w`*, 즉 각 채널 k에 대한 importance weight 뿐이었습니다. (이것을 알아내는 것이 핵심)
>
> 위에서 보았던 **<u>Saliency</u>**를 Backprop으로 구했던 방법과 비슷하지만 응용해서 수행합니다.
>
> 여기서 weight는 기존의 weight와 조금 다른 개념이기 때문에 alpha라고 합니다.
>
> - 기존의 Saliency test는 입력영상까지 backprop했지만, 여기에서는 원하는 Activation map(즉, 특정 conv 층)까지만 역전파를 수행합니다.



> **수식설명**
>
> $\alpha_k^c :\ C 클래스를\ 판별하는\ 것에\ 있어서\ \\
> k번째\ feature\ map의\ 중요성을\ 나타내주는\ 가중치이다$
>
> $y^c\ : 현재\ task에서\ 해석하고\ 싶은\ 결과\ (class에\ 대한\ score)$
>
> $A_{ij}^k\ :\ Conv\ Layer의\ k번째\ feature\ map$
>
> $\frac{\part{y^c}}{\part A_{ij}^k}:\ back\ progagation의\ gradients$
>
> $Z: i×j크기의\ 모든\ pixel\ 수\ (receptive\ field의\ 크기)$
>
> $\frac{1}{Z} \sum_i \sum_j \frac{\part{y^c}}{\part A_{ij}^k}
> :\ Global\ average\ pooling$
>
> <img src="Lecture6_CNN%20visualization.assets/img39.png" alt="image-20210313060446760" style="zoom: 67%;" />

> 
>
> - 클래스 c에 대한 스코어 값  y^c가 나오면 이로부터 Loss를 구합니다.
> - 이것에 대한 역전파를 통해  각 채널(c)과 피쳐맵(k번째)에 대한 weight (알파 {k} {c})가 각각 구해집니다.
>   - 알파를 구하는 과정에서 역전파를 통해 gradient를 사용하면 결과에 영향을 많이 미친 부분에 중요도를 더 높게 줄 수가 있는 것임.
> - 새로이 구한 weight(알파)와 Activation map A^k를 선형결합하여 ReLU를 적용합니다.
>   - 양수값만 사용하겠다는 뜻
> - 이를 히트맵으로 표현하면 Grad-CAM이 됩니다.
>
> ![image-20210313200109764](Lecture6_CNN%20visualization.assets/img40.png)
>
> 
>
> ※ 위 그림을 예시로 설명해보겠습니다.
>
> A가 마지막 fully connected layer의 Featur Map이고 **`사람`** 클래스를 검출해야 하는 TASK를 수행한다고 가정하면,
>
> 1. A heatmap의 어떤 pixel 값을 높게 줬더니 '사람' class의 score 값 (y)이 크게 향상 된 경우(변화율 증가) 
>    - => 그곳에 사람이 있다고 생각되서 alpha 값을 크게주고
> 2. A heatmap의 어떤 pixel 값을 높게 줬더니 '사람' class의 score 값 (y)이 낮게 향상 된 경우(변화율 감소)
>    - => 그곳에 사람이 없다고 생각되서 alpha 값을 낮게 줍니다.
> 3. 그리고는 ReLU를 사용해서 선형 결합을 하게 됩니다. (ReLU를 사용했기 떄문에 양수값만 사용하게 됩니다.)
>
> 4. 여기서 A heatmap의 α를 계수로 사용해, "alpha값이 강한 곳 * 물체 위치 (A feature map에서 강하게 나온 값)"은 강한색으로 나타냅니다.
>
> 
>
> 이와 같은 과정처럼 기존 Pretrained된 모델의 아키텍쳐를 따로 변경할 필요가 없기 때문에, 영상인식 Task에 한정될 필요 없이 거의 모든 Task에 사용되는 것을 볼 수 있습니다. (Image Classification, Image Captioning, Visual QA 등)
>
> <img src="Lecture6_CNN%20visualization.assets/img41.png" alt="image-20210313204235356" style="zoom:80%;" />





> **Guided Grad-CAM**
>
> 위 그림에는 Guided Grad-CAM이라고 되어있는데 이는 사실 Grad-CAM과 Guided Backprop을 결합하여 사용할 수 있기 때문에 합쳐놓은 것이라고 보면 된다.
>
> - Grad-CAM은 rough하고 smooth한 특성을 가지고 있고, (히트맵 보면 뭉뚱그려서 이 부분쯤이 중요하다~ 영역을 나타내주고 있음 : Smooth)
> - Guided Backprop은 sharp하지만 class 구분성이 조금 떨어지므로, (Class 구분성이 떨어져서 전반적으로 점이 몇개 없음. 즉,히트맵을 보면 sparse하게 픽셀 몇 부분만 찍어놓았음 : Sharp)
> - 이 두 개를 결합한 Guided Grad-CAM을 이용하면 해당 클래스에 대해 명확한 인식을 가지면서도 sharp하게 모양을 잡아낼 수 있다.



---

#### SCOUTER

최근에는 Grad-CAM을 좀 더 개선해서, "이 영상을 무엇으로 판단했느냐" 뿐만 아니라 "이 영상을 왜 그렇게 판단했느냐"까지 비교대조해볼 수 있는 SCOUTER 방법도 제안되었습니다.

- 다음은 왜 7인지, 왜 1이 아닌지, 왜 2가 아닌지 분석해본다.
- 어느 부분때문에 7로 판단했고, 어느 부분 때문에 1이 아니라고 판단했는지 등을 나타내주고 있다.

<img src="Lecture6_CNN%20visualization.assets/img42.png" alt="image-20210313204300593" style="zoom:80%;" />



#### +추가적으로 알아둘 부분 (생성모델에 응용)

해석을 위한 방법들을 위에서 배워보았는데, 이 방법을 잘 응용하면 다음 예시처럼(GAN dissection) 생성모델도 어떤 히든노드가 물체의 어떤 부분을 담당하는지와 어떤 부분을 생성하는데 Contribution을 했는지 해석을 할 수 있기 때문에 유저가 그 부분을 수정해서 사용할 수가 있습니다.

- 만약 우리가 문(door)과 관련된 히든 노드(채널)를 찾아놨고, GAN으로 생성된 이미지에 문이 없을 경우
- 문(door)의 위치에 해당하는 공간에다가 door를 담당하는 채널(hidden node)을 집어넣는, 그걸로 마스킹을 해주는 식으로 유저가 수정할 수 있다는 것이다.

<img src="Lecture6_CNN%20visualization.assets/img43.png" alt="image-20210313204321107" style="zoom: 67%;" />





아래는 앞선 내용 중 기억해야 할 부분, 궁금할 만한 부분을 되짚어보는 Q&A 부분입니다.

---

### +되짚어보기 (Q&A)

###### Q1) CAM에서 (GAP) Global Average Pooling이 어떻게 공간정보를 가지고 있게 되는 걸까?

- 일단 GAP가 공간정보(채널, 픽셀 둘다?)를 모두 가지고 하긴 어려운 것 같음.
- 채널 정보를 각각 들고 있다는 관점에선 공간정보를 가지고 있다는 표현이 맞을듯하다.
-  픽셀 정보는 CAM이 가지고 있는 것임.(GAP 적용되기 전 부분이 x,y 픽셀 정보를 가지고 있는 것임.)
- 아래에서 그림을 보면서 GAP가 채널정보를 어케 들고 있는지 추가적으로 자세히 설명해보겠다.

<img src="Lecture6_CNN%20visualization.assets/img44.png" alt="image-20210312002010388" style="zoom: 50%;" />



- 만약 Global Average Pooling의 Input으로서 C개의 채널의 feature map이 들어온다면, 각 채널에 Average Pooling을 통해 1x1 feature map를 C개 만들고 이를 concat하는 것이다.

- 결국 GAP의 결과는 C개의 각 채널의 정보를 1개의 픽셀(Average)로 표현하여 C개만큼 가지고 있는 것이다.
  - GAP는 각 채널의 대푯값(분포)만 뽑은 다음 concat하여 이렇게 만들어진 Flatten Vector를 FC layer에서 이용해 Task를 수행합니다
  - 이 과정에서 틀렸을 때의 loss가 발생할 것이고 이를 통해 여지껏 그랬던 것처럼 역전파 과정을 수행합니다
  - 위 그림에선 n개의 채널 분포값 중 어느 것이 중요한지를 loss에서 비롯된 역전파를 거쳐 가중치가 업데이트되겠죠.
  - 마치 해당 Task를 수행하는 것에 있어서 어느 채널이 중요한지 Attention 메커니즘을 수행하는 것이라고 볼 수도 있겠습니다.



```
말로 설명하기보단, 이제 위의 그림을 다시 한번 살펴보면서 GAP결과를 좀 더 Feature의 관점으로 이해해보자.

GAP 이전의 n개의 채널을 각각 압축시켜 n개의 정보들(Flatten Vector : w1, ... wn)로 만들고, 이를 FC Layer에 넣어 Task를 수행하는 정보가 된 것이다.

- 예를 들어, mnist의 경우에 최종 출력이 (1, 1, 10)이고 각 배열 안에 각 숫자에 대한 softmax 값이 들어있습니다.


그냥 이전부터 CNN에서 하던 flatten 이후 FC layer를 통과시키는 것과 과정이 거의 동일한 것이다 (과정에서 Flatten 대신 1x1 C랑 GAP를 사용했을 뿐)
```







###### Q2) 1x1 kernel size 로 컨볼루션하는데 왜 Fully Convoultion 결과값이 1x1xdimension 이 되는거에요?

> 어떤 feature map size가 W x H x C라고 가정해보자.
>
> 1. 이 때 해당 layer에서 1 x 1 convolution을 쓰면 (W, H, number of filter)이 됩니다, (filter 수는 우리가 조정가능함, 최종적으로 원하는 output의 class 개수로 설정하면 되는거임)
> 2. Global Average Pooling을 쓰면 (1, 1, C)가 됩니다.
>
> - 참고 : 그냥 flatten을 쓰면 (1, W x H x C)가 됩니다.
>   - 이 경우에는 우리가 최종적으로 원하는 아웃풋을 뽑고 싶으면, (원하는 결과 차원수가 있다면) nn.Linear() 등과 같이 output channel를 class 갯수만큼 줄여주는 과정을 추가해야 한다.
>
> 
>
> 주의할 점 : GAP 부분은 flatten처럼 펼쳐주는 개념이 아니라 fc layer처럼 (1, 1, channel) 꼴로 만드는 것
>
> 
>
> 결론 : Fully Connected Layer를 대체하려면, (1x1 convolution과 Global Average Pooling을 함께 사용하며, 이것을 Fully Convolution Layer라고 부르는 것이다.)



```
※ 요약정리)

만약 내가 Input size에 상관없이 FC layer처럼 내가 원하는 shape의 출력을 얻고 싶다면,
1. 마지막 conv layer보다 먼저 1 x 1 convolution을 사용해서 내가 원하는 output channel만큼 줄인 후,
2. gap를 사용해 fc layer처럼 (1, 1, channel) 꼴로 만듭니다. - 각 채널별로 해당 채널이 모든 픽셀들의 평균
3. 그 후 내가 원하는 Task를 수행하기 위해 이를 다음 layer에 집어넣어주면 됨. (분류라면 Softmax 등)

MNIST 데이터셋의 경우 Class가 10개잖아.
- 그러면 마지막 부분에서 1x1 Conv로 내가 원하는 클래스 개수 10만큼 줄여주고
- Global Average Pooling 쓴 다음
- Softmax를 사용하면 되는거지.
```


