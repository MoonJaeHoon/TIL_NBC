# Conditional Generative Model



## 1. Conditional Generative Model

일반적인 generative model은 영상, sample을 생성할 수 있지만 조작은 불가능했습니다.

Conditional generative model은 조건(condition)d이 주어졌을 때 영상, sample을 생성하는 모델입니다. 따라서 user의 조작이 더 쉬어졌습니다.

 ![image-20210315002840715](Lecture8_cGAN (Conditional generative model).assets/image-20210315002840715.png)





> **GAN 개념 돌아보기 (아래 그림)**
>
> 참고 : <a href="../GAN 간단 정리/GAN_Concept.md" target="_blank">GAN 개념정리</a>



![image-20210315002940262](Lecture8_cGAN (Conditional generative model).assets/image-20210315002940262.png)



![image-20210315001927717](Lecture8_cGAN (Conditional generative model).assets/image-20210315001927717.png)

 

> GAN vs Conditional GAN

Conditional GAN이 기존 GAN과 다른점은 C라는 Conditional input을 넣어주는 부분이 다릅니다.

 

![image-20210315003132645](Lecture8_cGAN (Conditional generative model).assets/image-20210315003132645.png)

> 응용 작업들

이미지를 새로운 이미지로 만드는 작업 (사진→수채화그림, super resolution, 등의 style transfer)

게임 테마 제작. 인게임의 새로운 테마를 제작할 때 디자인 비용이 대폭 감소할 것이다.

<img src="Lecture8_cGAN (Conditional generative model).assets/image-20210315003602112.png" alt="image-20210315003602112" style="zoom:80%;" />

<img src="Lecture8_cGAN (Conditional generative model).assets/image-20210315003621002.png" alt="image-20210315003621002" style="zoom: 67%;" />



> Super Resolution (저해상도 => 고해상도)

Conditional GAN의 예시에는 Super resolution 기법이 있습니다.

- **input :** Low resolution image (저해상도 image)
- **output :** high resoulution image (고해상도 image)

![image-20210315003243566](Lecture8_cGAN (Conditional generative model).assets/image-20210315003243566.png)

> Super Resolution GAN의 구조

구조는 다음과 같습니다.

입력으로는 Low resolution image가 주어지게 되고, Generator는 high resolution image를 생성합니다.

Ground Truth로는 실제의 high resolution image를 주어서 Discrimitor가 현재 주어진 생성 image가 실제 high resolution image와 **비슷한 통계적 특성을 갖는지** 확인합니다.(Discriminator)

 ![image-20210315004123913](Lecture8_cGAN (Conditional generative model).assets/image-20210315004123913.png)



 

> Regression vs CGAN

 원래는 Regression을 사용했었는데 Discrimitor가 아니라 MAE / MSE 등의 평가지표를 사용했습니다. **<u>이는 문제점이 있어서</u>** GAN 메커니즘 등장 이후로는 GAN을 거의 모두가 이용한다.

![image-20210315003822617](Lecture8_cGAN (Conditional generative model).assets/image-20210315003822617.png)



> 어떤 문제점일까?

Regression을 사용하면 해상도가 높아지긴 하겠지만 sharp한 영상 대신, blur처리가 된 영상을 많이 생성하게 됩니다.



> 문제점의 이유

MAE나 MSE 등의 평균을 이용한 Loss를 이용하는 경우의 문제점에 대해 다음 그림을 보며 생각해보자.

입력 이미지의 해상도를 높였을 때와 비슷한 <font color='red'>**ground truth**</font>들이 아래와 같이 분포한다고 하면, Loss가 가장 작은 정답을 예측하려면 모든 ground truth와 동떨어진 <font color='skyblue'>**중간 지점의 뭉개진 이미지**</font>를 구하게 된다.

![image-20210315004439232](Lecture8_cGAN (Conditional generative model).assets/image-20210315004439232.png)

반면에 GAN의 Discrimitor는 그 전에 봤었던 real data와 생성된 data를 구분 못하게 하는 것만 목적이므로 가장 <font color='khaki'>**비슷한 real data만 따라하게**</font> 됩니다. 그러면 Discrimitor의 loss가 낮아지게 됩니다.

 

> 쉬운 예시

위의 문제점(평균값 등을 지표로 해선 안되는 이유)을 이해하기 쉽게 예를 들어보겠다.

다음 그림과 같이 검정색과 하얀색 이미지(Real Image)가 있는 경우가 바로 그 예시일 것이다.

![image-20210315005418231](Lecture8_cGAN (Conditional generative model).assets/image-20210315005418231.png)

 

 

## 2. Image translation

 

Image translation은 한 Image style을 다른 image style로 변환하는 방법입니다.

 ![image-20210315005602108](Lecture8_cGAN (Conditional generative model).assets/image-20210315005602108.png)

 

 

### 2.1 Pix2Pix

Pix2Pix는 Image translation이라는 task를 CNN 구조를 이용해서 학습기반으로 나온 첫 번째 사례입니다.

Pix2Pix는 Loss function을 다음과 같이 정의했습니다.

 ![image-20210315005635555](Lecture8_cGAN (Conditional generative model).assets/image-20210315005635555.png)



- **L_{L1}(G)**는 MAE Loss를 의미합니다.

- MAE Loss가 Blury한 image를 생성하긴 하지만 적당한 guide로 사용하기에는 좋습니다.

- **LcGAN(G,D)**는 GAN Loss를 의미합니다.

- GAN Loss를 사용해서 realistic한 출력을 만들도록 유도해줍니다.



여기서 MAE Loss를 사용한 이유는 MAE Loss에서 y는 ground truth인데 x라는 입력을 넣었을 때 기대하고 있는 출력 pair를 가지고 있는 데이터에 대해서 학습을 진행합니다.

하지만 GAN Loss는 ground truth와 직접 비교하지는 않습니다. D(x,y)에서 x와 y를 **독립적**으로 dicrimitor해서 real이냐 fake이냐만 구분하게 됩니다.

 

>  Pix2Pix의 GAN Loss vs 일반적인 GAN Loss

 GAN Loss에서 G(x,z)와 같이 z만 들어간 것이 아닌 x까지 들어갔다는 점입니다.

![image-20210315010121702](Lecture8_cGAN (Conditional generative model).assets/image-20210315010121702.png)



L1 loss와 GAN loss를 같이 사용하게 되면 아래 그림과 같이 스타일이 유지되면서 sharp하게 사진이 나오는 것을 알 수 있습니다.

![image-20210315010331585](Lecture8_cGAN (Conditional generative model).assets/image-20210315010331585.png)



 

### 2.2 CycleGAN

Pix2Pix는 **pairwise data**가 필요합니다.

즉, 예를 들어서 sketch data (xi)와 일반 data (yi)와의 관계를 학습하는 pair 형태가 필요합니다.

문제는 pair data를 얻는게 어렵거나 불가능하다 (하나하나 pair를 해주기도 작업량이 너무 많기 때문이다.)

 따라서 CycleGAN은 이 점을 해결하기 위해 Pair되지 않은 두 개의 묶음들로 학습을 시킵니다. (unpaired 방법)

- 아래 그림처럼, X라는 style의 data와 Y라는 style의 data는 서로 직접적인 연관이나 대응관계 없이 주어졌을 때 활용하는 방법입니다. 

![image-20210315010602704](Lecture8_cGAN (Conditional generative model).assets/image-20210315010602704.png)



※ CycleGAN은 non-pairwise dataset만으로 image translation이 가능하도록 만들었습니다.

다음은 원리를 보여주는 간단한 예시들

- 원본 사진을 모네스타일의 이미지로 변경한 후, 다시 원본스타일의 이미지로 변경해서, 원본 사진과 비교한다.

<img src="Lecture8_cGAN (Conditional generative model).assets/image-20210315010916710.png" alt="image-20210315010916710" style="zoom:150%;" />

- 원본 사진을 일반 말의 이미지로 변경한 후, 다시 원본스타일의 이미지로 변경해서, 원본 사진과 비교한다.

<img src="Lecture8_cGAN (Conditional generative model).assets/image-20210315010930783.png" alt="image-20210315010930783" style="zoom:150%;" />



> 수식 측면에서 살펴보겠습니다.

CycleGAN에는 GAN Loss + **Cycle-consistency loss**가 추가되었습니다.

 ![image-20210315011108322](Lecture8_cGAN (Conditional generative model).assets/image-20210315011108322.png)



여기 Loss 이름 앞에 붙은 `Cycle`이라는 단어처럼 방향성이 존재합니다(X -> Y, Y -> X 등). 따라서 이 두 방향을 동시에 학습을 진행합니다.

**즉, Cycle-consistency loss는 원본 image를 통해 translation된 결과를 출력하고, 출력된 결과를 다시 원본 image와 비교했을 때 동일해야한다는 loss입니다.**



> GAN Loss의 계산은 다음과 같이 진행됩니다.

1. X -> Y를 G (generator)를 통해서 생성을 합니다.
2. D_Y는 Y를 dicrimitor합니다.
3. Y -> X로 F (또다른 generator)를 통해서 생성을 합니다.
4. D_X는 X를 dicrimitor합니다.

![image-20210315011309169](Lecture8_cGAN (Conditional generative model).assets/image-20210315011309169.png)

 

하지만 GAN Loss만 사용하면 **Mode Collapse**라는 문제가 발생합니다. 

Mode Collapse : input에 상관 없이 하나의 output만 계속 출력하는 형태로 학습이 되는 것

 

따라서 **Cycle-consistency loss**를 사용하게 됩니다.

X -> Y로 가고 다시 Y -> X로 갈 때 **`X`**와 **`다시 돌아온 X`**에서 차이가 있으면 안된다는 것입니다.

즉, `x`가 `x^hat`과 동일해야한다는 것입니다. (즉,  self-supervised)

![image-20210315011925245](Lecture8_cGAN (Conditional generative model).assets/image-20210315011925245.png)

 

### 2.3 Perceptual loss

Perceptual loss는 high quality output을 만들기 위한 방법 중 하나입니다.

- Adversarial Loss (GAN Loss)
  - train하기 어려움
  - pre-trained가 필요하지 않고, Generator와 Discriminator가 균형을 맞추게 됨



- Perceptual Loss
  -  train하기 쉽고, coding하기 쉬움
  -  pre-trained network를 사용해야함



> Peceptual loss를 사용해서 Image Transform Net을 구축하는 과정

- **Image Transform Net :** input image가 주어지면 image를 transform해서 출력 (하나의 input image가 들어오면 어떤 **<u>하나의 style</u>**로 바꾸는 역할을 하게 되는 네트워크이다.)

- **y^hat:** Image tranform net의 출력

- **Loss Network :** 학습된 Loss를 측정하기 위해서 **image classification network**를 사용합니다. (전형적으로, pre-trained된 VGG-16 Image net이 사용됨).
- 이 **네트워크로 중간중간의 feature를 뽑습니다.**
- Backpropagation을 할 때 중간중간의 feature들로부터 gradient를 구해서 `y^hat`이 온 방향으로 돌아가면 `y^hat`을 업데이트해주며 학습합니다.
  - 이때, Loss Network model 자체는 <font color='red'>update하지 않습니다</font>.(Pre-trained로 가져와서 fixed로서 씀)
- Style Target과 Content Target을 통해 학습하게 됩니다.

 ![image-20210315012442800](Lecture8_cGAN (Conditional generative model).assets/image-20210315012442800.png)

 

> **Feature Reconstruction Loss 설명**

**Feature reconsturction loss**는 두 가지 (style target : y^s, content target : y^c)로부터 각각 Input image X를 비교하여 계산된 2개의 loss(Contents reconstruction loss, Style reconstruction loss)을 사용하여 계산됩니다.

 

1. **Contents reconstruction loss**

   - **Contents target**은 Transformed Image가 Contents를 제대로 유지하고있는지 그것을 확인해주는 Loss입니다. 즉, contents target로서 **`y^c`**가 들어가고, input으로서 **`원본 X`**가 들어가고 둘을 비교해 loss 계산하게 되는 것임.
   - 이 둘을 단순히 그냥 비교하는 것이 아니라, **`X`를 Pre-trained VGG에 넣어서 feature map** 뽑고, **`y^c`도 Pre-trained VGG에 넣고 뽑은 feature map** 사이를 비교해서 L2 Loss를 구하게 됩니다.
   - Loss로부터 역전파를 하여 `y^hat`을 업데이트하게 됩니다. (정확히는 Image Transform Net의 f_{W}가 바뀌게 되는 것.)

   ![image-20210315023238717](Lecture8_cGAN (Conditional generative model).assets/image-20210315023238717.png)

 

2. **Style reconstruction loss**
   - **Style Target**에는 **우리가 변환하고 싶은 style image를 input**으로 넣습니다.
   - **`원본 X`를 VGG에 넣어서 feature map** 뽑고, 변경하고 싶은 타겟 **style image `y^s`를 VGG에 넣고 뽑은 feature map**을 비교하는데 위와는 다릅니다.
   - 여기에서 바로 Feature map끼리 비교를 해서 loss를 구하는 것이 아니라, Style이라는 정보를 담기 위해서 Style을 design해준 뒤 비교를 합니다.
   - 이렇게 스타일 정보를 담기 위해 만들어진 것이 바로 **Gram matrices**입니다.



> **Gram matrices 설명**
>
> - **Gram matrices**는 일반적으로 feature map에서 공간적인 정보(spatial information)의 통계적(statistical) 특징을 담을려고 디자인되었습니다.
>
> - Gram matrices는 channel(c) x channel(c) 형태로 구성되어 있는데 shape을 이렇게 만드는 방법은 다음과 같습니다.
>
> - 예를 들어서, 아래 그림과 같이 Feature Maps가 뽑혔다고 생각해봅시다. (W X H, channel : c)
>
>   ![image-20210315030405614](Lecture8_cGAN (Conditional generative model).assets/image-20210315030405614.png)
>
>   - 스타일을 고려하고 싶다는 뜻은 다시 생각해보면, 각 위치마다 (픽셀마다라고 이해했다) 정보가 다른 것을 계산해서 고려하는 것이 아니라 **영상의 전체적인 Style을 고려하고 싶다**는 뜻일 것입니다.
>
>   - 따라서, 공간에 대한 정보를 쪼끔 없애주는 과정을 거칩니다 (이는 당연히 Pooling을 통해서 공간에 따른 정보를 없애고 함축시킬 수가 있겠죠?)
>
>   - 즉, 원래는 C X H X W였던 feature map을 C×(H⋅W)형태로 reshape해주는 과정을 해줍니다.
>
>   - reshape한 C×(H⋅W) feature map으로부터 내적연산을 하게 되면 다음과 같이 CxC matrix가 나오게 될 것입니다.
>
>   - 이렇게 내적연산으로부터 나오는 gram matrix내의 value들의 기하학적인 의미는 각 채널별로 가지고 있는 특징의 동시등장확률을 나타냅니다.(빈도 기반 유사도 측정)
>
>     - 예를 들어, 채널1이 내포하고 있는 style 특징이 원이 등장하는 부분을 캐치하고 있는 것이고, 채널2가 내포하고 있는 style 특징이 가로선이 등장하는 부분을 캐치하고 있는 것이라고 해보겠습니다.
>     - 그리고 직접 그린 다음 그림처럼 채널1과 채널2 사이의 내적이 나타내는 element가 높다면 이 영상은 style 적으로 원과 가로선이 같이 나타나는 빈도가 높다는 것입니다.
>
>     <img src="Lecture8_cGAN (Conditional generative model).assets/image-20210315030905524.png" alt="image-20210315030905524" style="zoom: 67%;" />
>
> - 이렇게 나온 결과 matrix를 gram matrix라고 하며, 하나의 feature map에서만 이렇게 구하는 것이 아니라 여러개의 feature map으로부터 여러개의 gram matrix를 구할 수 있을 것이고 이를 gram matrices라고 부른다.
>
> - 각각의 level이 다른 feature map으로부터 나온 gram matrix가 있을 것이기 때문에 따로따로 서로 다른 level에서의 loss를 계산하게 된다.
>
> 
>
> **※ 결국, Transformed image의 gram matrix는 "staticstic한 style들을 저장한 style target의 gram matrix"를 닮아가게 됩니다.** 