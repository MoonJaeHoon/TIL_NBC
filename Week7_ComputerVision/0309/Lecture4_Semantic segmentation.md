# Semantic Segmentaion



## 1.1 Semantic Segmentaion이란?

Semantic Segmentaion은 영상단위로 어떤 분류를 하는 것이 아니라 픽셀단위로 어떤 카테고리인지 분류를 한다.



<img src="Lecture4_Semantic segmentation.assets/image-20210309211327174.png" alt="image-20210309211327174" style="zoom:67%;" />

- 주의할 점 : 자동차가 여러개 있더라도 모두 같은 카테고리(자동차), 사람이 여러명 있더라도 같은 카테고리(사람)이라고 분류를 하는 Task이다.
- 목적 자체가 그렇게 설계되었음



> 참고: Instance Segmentaion은 인스턴스까지 구분을 해준다.



## 1.2 Semantic Segmentation 적용분야

<img src="Lecture4_Semantic segmentation.assets/image-20210309211505745.png" alt="image-20210309211505745" style="zoom:67%;" />



> Computational photography

어떤 물체를 바꾸거나 수정할 수가 있음 (포토샵 등을 이용해)

다음과 같이 인물집중 사진으로서 주변환경 흐리게 하는 것도 그 예시

배경이나 인물을 아예 다른 배경화면, 또는 인물로 바꾸거나 하는 것도 그 예시

<img src="Lecture4_Semantic segmentation.assets/image-20210309211737947.png" alt="image-20210309211737947" style="zoom:50%;" />



## 2. Semantic Segmentation Architectures

Semantic Segmentation Model에는 여러가지가 있다.



### 2.1 Fully Convolutional Network

Semantic Segmentation를 위한 첫번째의 end-to-end 뉴럴넷이다 

>  end-to-end : 우리가 원하는 Task를 단 하나의 아키텍쳐로 처음부터 끝까지 수행할 수 있는 방법을 말하는 것 같다.
>
> 예를 들어서, Alexnet의 경우 결과에서 FC layer Part에서 Flattening Vector를 만들기 때문에 (벡터화시키기 때문에) 입력해상도가 포함되지 않으면 학습되어진 FC layer의 정보를 사용할 수가 없는 한계가 존재했다.
>
> 입력해상도가 바뀌는 순간 컨볼루션 레이어의 activation map의 dimension이 달라지게 되고 이를 flattening하면 길이가 아예 달라지게 되기 때문이다.

<img src="Lecture4_Semantic segmentation.assets/image-20210309212443424.png" alt="image-20210309212443424" style="zoom:67%;" />

