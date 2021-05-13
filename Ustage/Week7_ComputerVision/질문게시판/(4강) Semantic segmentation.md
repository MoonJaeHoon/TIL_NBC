# 1. Fully Convolution Network관련 질문 드립니다.

안녕하세요 semantic segmentation 공부를 하다 궁금한게 생겨 질문 드립니다.

FCN-16s 와 FCN-8s에서는 pool4와 pool3과 같이 pooling 이후 이미지를 사용하는 것 같은데, conv4나 conv5와 같이 convolution이후 결과 대신 pooling이후 결과를 사용하는 이유가 무엇인가요?

<img src="(4강) Semantic segmentation.assets/mceclip0.png" alt="img" style="zoom:67%;" />





> 어떤 intermediate feature map에서 정보를 끌어올지는 정해진 사항이 없습니다. 모델 종류마다 다를 뿐만 아니라 정해진 규칙이 없어서 FCN에서는 저렇게 레이어를 구성했나보다 정도로 정리하셔도 될 것 같습니다 :)





# 2. 이미지의 사이즈를 줄였다가 늘리는 과정을 왜 하는지 궁금합니다.

semantic segmentation을 공부하다가 이해가 안되는 부분은 이미지를 pooling을 통해 width와 height를 줄이고 줄여서 작게 만들면서 channel의 수를 늘리고 다시 원래 크기로 복원합니다.

여기서 왜 줄였다가 늘리는지 잘 모르겠습니다.

픽셀마다 클래스를 부여하고 싶다면 입력의 사이즈 W x H를 유지한채 channel의 수만 조절하여 최종 출력값이 W x H x num_class 로 만들면 안되나요?



또 다른 질문은 신경망을 공부하다 보면 모델에서 차원(혹은 채널)을 늘렸다가 줄였다가를 반복하는 경우를 자주 봤습니다.

왜 차원 혹은 채널의 수를 늘렸다가 줄였다가를 반복하는지 궁금합니다.

제 생각은 늘리는 경우, 숨겨진 정보를 얻기 위함이고 줄이는 경우는 1x1 conv 처럼 계산량을 줄이기 위함으로 생각됩니다.





> 안녕하세요 조교 이주용입니다.
>
> 우선 subsampling의 과정을 통해 차원을 줄여 연산량에 이점을 얻는 것 외에도, optimize해야 하는 parameter의 수를 줄여 overfitting을 방지하거나 불변량 검출을 의도하여 성능 향상 등의 효과를 얻을 수 있습니다. semantic segmentation에서 (1) 먼저 convolution & pooling layer들의 다양한 이점들을 활용해 feature를 정보를 추출해내면, (2) (보통 줄어든) 차원을 다시 원본 이미지에 맞춰주는 것이 필요합니다. 이 때 upsampling을 쓰는데, 저해상도 feature에서 고해상도 feature(혹은 이미지)를 얻기 위해 적용한다고 생각하셔도 되고, 추출해낸 feature가 원본 이미지 상에서 어떤 spatial 정보를 지닌 채 decoding되는지 확인한다고 생각해도 될 것 같습니다.
>
> 말씀하신 대로 WxH를 유지한 채, 진행할 수도 있습니다. 이런 경우 다른 이점들이 있겠지만, 여태 이야기했던 convolution & pooling 효과를 충분히 활용하기는 힘들 것 같네요.



**+추가질문**

\1. 답변해주신 내용에서 subsampling으로 "optimize해야 하는 parameter의 수를 줄여 overfitting을 방지"하는 효과가 있다고 말씀해주신 부분에 대해서 궁금한 점이 있습니다. W와 H를 유지하면서 convolution layer를 적용하려면 아마 패딩과 스트라이드를 적절히 이용하면서 해야할 것 같은데요. 그렇게 하더라도 파라미터 수가 늘어나지는 않으니 괜찮지 않을까요?

 

\2. 답변해 주신 내용에서 불변량 검출에 대해서 좀 더 자세히 설명해주시면 감사 하겠습니다. 어떤 부분을 불변량이라고 할 수 있는지를 잘 모르겠습니다.



> 추가질문에 대해서 답변 드릴게요.
>
> 1) 말씀해주신 경우에는 parameter수를 줄이는 효과는 없는게 맞습니다. 좀 더 일반적인 관점에서 subsampling의 효과를 이야기한 것으로 이해해주시면 됩니다.
>
> 2) 가령, 우리가 원하는 정보가 어떤 특정 원형의 물체가 어디에 있는지가 아니라, 그냥 있는지 없는지 여부라고 생각해봅시다. 이런 경우, spatial한 정보들을 전부 유지시켜주는 것보다 불변성을 활용하는 것이 더 나을 수 있습니다. (원형의 물체의 존재여부는 픽셀이 조금 이동하더라도 변하지 않을 것이기 때문에, 해당 용어를 썼습니다.)





> 추가로 답변을 더 드리면 feature map의 크기를 점점 줄이는 과정이 parameter 수를 줄이기 위함도 있지만 filter의 receptive field를 키우기 위함도 있습니다.
>
>  
>
> 네트워크 앞단에 위치한 conv layers의 경우, input image에서 굉장히 local한 영역만 receptive field로 가지게 되어 결국 이러한 field는 비교적 detail하고 local한 정보를 잘 추출해내게 됩니다. 이와 달리 뒷단에 위치한 conv layers의 경우, 동일한 filter size더라도 크기가 줄어든 feature map에 대해 연산이 진행되기 때문에 input image로 따지면 훨씬 global한 영역을 receptive field로 가지게 됩니다. 이러한 filter들은 물체의 전반적인 형태나 추상적인 feature를 얻어내는 것에 특화됩니다.
>
>  
>
> 결국 feature map size를 점점 줄여나가면 앞쪽의 filter들은 local features를 잘 얻어낼 수 있고 뒤쪽의 filters는 global features를 잘 얻어낼 수 있기 때문에 network 전반적으로 보면 local한 정보와 global한 정보를 모두 잘 고려하여 풀고자 하는 task를 수행할 수 있는 feature extractor가 완성되게 됩니다.





# 3. U net구조에 대해서 질문드립니다.

안녕하세요, U net구조에 대해서 질문드립니다. 

다음과 같이 contracting path의 일부가 Expanding path에 전달되서 concatenate되는 부분에서,

이미지 사이즈가 맞지 않아 그림과 같이 일부만 전달을 하기위해 crop을 하는 것으로 이해했습니다. (appendix)

[<img src="(4강) Semantic segmentation.assets/IMG_5F6179C8EB53-1.jpeg" alt="img" style="zoom: 50%;" />](https://cphinf.pstatic.net/mooc/20210309_36/1615299191896ge3sn_JPEG/IMG_5F6179C8EB53-1.jpeg)

[![img](https://cphinf.pstatic.net/mooc/20210309_292/1615299235783ugYvS_JPEG/IMG_2CB90050850D-1.jpeg)](https://cphinf.pstatic.net/mooc/20210309_292/1615299235783ugYvS_JPEG/IMG_2CB90050850D-1.jpeg)

1) crop을 하는 방법에 대해 자세히 알 수 있을까요? 랜덤으로 crop을 해서 전달하나요?



2) 또한, 그렇게 crop을 해서 전달이 되면, concatenate할 때, 같은 (n,m)위치에 있는 픽셀이 내포하고 있는 dataset 영역이 expanding path의 feature map과 contracting path의 featue map이 서로 다를 것 같습니다. 그러면 오히려 노이즈가 되지 않을까요?



> U-net 구조에서 crop은 중앙 부분을 crop합니다. 논문에서는 "The cropping is necessary due to the loss of border pixels in every convolution" 라고 소개되어 있습니다. 제 생각에는, 결국 convolution 연산 후에 날아간 border 정보들은 중간으로 집중되어 있을 것이기 때문에 중앙 부분을 crop하는 선택이 한 쪽으로 치우친 것 보다 더 나은 선택인게 맞는것 같습니다. 이렇게 하면, 지적하신대로 같은 픽셀에서 비슷한 정보를 유지하는 효과도 있는것 같네요.



**+추가질문**

답변해주신 내용에 대해서 질문이 있습니다.

혹시 border부분의 정보가 중앙으로 집중하는 이유에 대해서 설명해주시면 감사하겠습니다. 

convolution시에 input의 border 부분의 정보는 output의 border 부분에 mapping되지 않나요? output의 border 부분의 값들을 계산하는데에는 input의 border부분의 값들이 사용 되므로 그렇게 되지 안을까 하고 생각됩니다.



> 답변 드립니다.
>
> 말씀하신 내용 맞습니다. 중간으로 집중이라는 표현이 좀 애매하긴 한데, crop되는 아주 border 정보들도 crop되지 않을 (비교적 중간) 부분에 포함되어 있을거라서 아주 border 부분을 버리는게 나을 것 같다는 생각이었습니다.