





> (GAP) global average pooling이 어떻게 공간정보를 가지고 있게 되는 걸까?

<img src="Lecture6_CNN visualization.assets/image-20210312002010388.png" alt="image-20210312002010388" style="zoom: 50%;" />

- FC는 공간정보가 없어지는 단점이 있어서 GAP나온 것이다.
- 예를 들어 Global Average Pooling의 Input으로서 C개의 채널의 feature map이 들어올 때, 각 채널에 Average Pooling을 통해 1x1 feature map를 C개 만들고 이를 concat하는 것이다.
- 결국 GAP의 결과는 각 채널의 정보를 1개의 픽셀(Average)로 표현하여 C개만큼 가지고 있는 것이다.
  - GAP는 각 채널의 대푯값(분포)만 뽑은 다음 concat하여 이렇게 만들어진 Flatten Vector를 FC layer에서 이용해 Task를 수행합니다
  - 이 과정에서 틀렸을 때의 loss가 발생할 것이고 이를 통해 여지껏 그랬던 것처럼 역전파 과정을 수행합니다
  - 위 그림에선 n개의 채널 분포값 중 어느 것이 중요한지를 loss에서 비롯된 역전파를 거쳐 가중치가 업데이트되겠죠
  - 마치 해당 Task를 수행하는 것에 있어서 어느 채널이 중요한지 Attention 메커니즘을 수행하는 것이라고 볼 수도 있겠습니다.



```
말로 설명하기보단, 이제 위의 그림을 보면서 GAP결과를 좀 더 Feature의 관점으로 이해해보자.

GAP 이전의 n개의 채널을 각각 압축시켜 n개의 정보들(Flatten Vector : w1, ... wn)로 만들고, 이를 FC Layer에 넣어 Task를 수행하는 정보가 된 것이다.

이전에 CNN에서 하던 flatten 이후 FC layer를 통과시키는 것과 과정이 거의 동일한 것이다 (Flatten 대신 GAP를 사용했을 뿐)
```







Q) 1x1 kernel size 로 컨볼루션하는데 왜 Fully Convoultion 결과값이 1x1xdimension 이 되는거에요?

> 어떤 feature map size가 W x H x C라고 가정해보자.
> 1. 이 때 해당 layer에서 1 x 1 convolution을 쓰면 (W, H, number of filter)이 됩니다, (filter 수는 우리가 조정가능함, 최종적으로 원하는 output의 class로 설정하면 되는거임)
> 2. Global Average Pooling을 쓰면 (1, 1, C)가 됩니다.
>
> - 참고 : 그냥 flatten을 쓰면 (1, W x H x C)가 됩니다.
>   - 이 경우에는 우리가 최종적으로 원하는 아웃풋을 뽑고 싶으면, (원하는 결과 차원수가 있다면) nn.Linear() 등과 같이 output channel를 class 갯수만큼 줄여주는 과정을 추가해야 한다.
>
> 
>
> 결론 : Fully Connected Layer를 대체하려면, (1x1 convolution과 Global Average Pooling을 함께 사용하며, 이것을 Fully Convolution Layer라고 부르는 것이다.)





만약 내가 Input size에 상관없이 FC layer처럼 내가 원하는 shape의 출력을 얻고 싶다면 마지막 conv layer에 1 x 1 convolution을 먼저 사용해서 내가 원하는 output channel만큼 줄인 후 gap를 사용해 fc layer처럼 (1, 1, channel) 꼴로 만듭니다. 예를들어 mnist의 경우에 최종 출력이 (1, 1, 10)이고 각 배열 안에 각 숫자에 대한 softmax 값이 들어있습니다.

gap으로 flatten처럼 펼쳐주는게 아니라 fc layer처럼 (1, 1, channel) 꼴로 만드는 것