# 1. Fully Convolution Network관련 질문 드립니다.

안녕하세요 semantic segmentation 공부를 하다 궁금한게 생겨 질문 드립니다.

FCN-16s 와 FCN-8s에서는 pool4와 pool3과 같이 pooling 이후 이미지를 사용하는 것 같은데, conv4나 conv5와 같이 convolution이후 결과 대신 pooling이후 결과를 사용하는 이유가 무엇인가요?

<img src="(4강) Semantic segmentation.assets/mceclip0.png" alt="img" style="zoom:67%;" />





> 어떤 intermediate feature map에서 정보를 끌어올지는 정해진 사항이 없습니다. 모델 종류마다 다를 뿐만 아니라 정해진 규칙이 없어서 FCN에서는 저렇게 레이어를 구성했나보다 정도로 정리하셔도 될 것 같습니다 :)