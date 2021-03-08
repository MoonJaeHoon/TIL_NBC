[→ Open in Slid](https://slid.cc/vdocs/4fd5b88d010e4079bc73f74786e98596)


---


&nbsp; 우리는 오감 중 특히 시각에 의존하여 사물을 바라보고 이해하며 살아가고 있습니다.<br>동일한 프로세스를 컴퓨터에 적용한 컴퓨터 비전입니다.<br><br>본 강의에서는 컴퓨터 비전 (CV)의 첫 시간으로 CV에 대해 짧게 소개하고, CV에서 가장 기본적인 task, image clasiification을 소개합니다.<br>Image Classification은 사진이 주어졌을 때&nbsp; 특정 카테고리로 분류하는 task입니다.<br><br>이번 강의에서는 먼저 기존의 머신러닝과 구분되는 딥러닝을 사용한 Image classification의 특징에 대해서 배웁니다.<br>다음으로 대표적인 CNN 모델인 AlexNet을 배우고 이에 대한 실습을 진행합니다.<br>끝으로 가장 유명한 classification 모델 중 하나인 VGGNet에 대해 배웁니다.<br>


Further Reading


- VGGNet :&nbsp;<a href="https://arxiv.org/pdf/1409.1556.pdf">https://arxiv.org/pdf/1409.1556.pdf</a>




![](https://slid-capture.s3.ap-northeast-2.amazonaws.com/public/capture_images/4fd5b88d010e4079bc73f74786e98596/6608f52c-9c28-4865-aa64-819765085a14.png "(1강) Image classification 1 image")


Computer Vision은 인간의 지능에 해당하는 Task를 수행할 수 있게 하는 것 (사고능력, 시각능력, 지각능력, 인지능력, 기억능력 등)




![](https://slid-capture.s3.ap-northeast-2.amazonaws.com/public/capture_images/4fd5b88d010e4079bc73f74786e98596/dda94e60-b7e2-4814-b900-edcb2aa2d838.png "(1강) Image classification 1 image")




![](https://slid-capture.s3.ap-northeast-2.amazonaws.com/public/capture_images/4fd5b88d010e4079bc73f74786e98596/02658931-cbd3-4b94-993f-938b1a44d538.png "(1강) Image classification 1 image")


 - 인간의 인지과정




![](https://slid-capture.s3.ap-northeast-2.amazonaws.com/public/capture_images/4fd5b88d010e4079bc73f74786e98596/16499be6-5136-46ad-a5dc-089110ce455c.png "(1강) Image classification 1 image")


 - 인간의 인지과정을 컴퓨터에 적용

![](https://slid-capture.s3.ap-northeast-2.amazonaws.com/public/capture_images/4fd5b88d010e4079bc73f74786e98596/54a074bb-2ba5-4e85-a0df-21b120daf905.png "(1강) Image classification 1 image")


 - Computer Graphics는 정보를 통해서 image를 그려내는 것
 - Computer Vision은 Image로부터 인지하는 과정


 - 두 가지가 Inverse 관계임










[![(1강) Image classification 1 video poster](https://slid-capture.s3.ap-northeast-2.amazonaws.com/public/clip_video_poster/4fd5b88d010e4079bc73f74786e98596/1fb12bcb-6318-4648-b32f-0983b1720777.png)](https://slid.cc/docs/4fd5b88d010e4079bc73f74786e98596)


 - 사람의 시각능력을 구현하는 것뿐만 아니라 이해, 인지능력까지 컴퓨터비전은 포함하려고함.

[![(1강) Image classification 1 video poster](https://slid-capture.s3.ap-northeast-2.amazonaws.com/public/clip_video_poster/4fd5b88d010e4079bc73f74786e98596/accfc3b9-f2e6-4723-8312-77bf9030cd2b.png)](https://slid.cc/docs/4fd5b88d010e4079bc73f74786e98596)


 - 사람의 시각능력도 불완전하다. (시력 이런 것 말고, 치명적으로 우리 눈이 저지르는 실수가 존재한다)




[![(1강) Image classification 1 video poster](https://slid-capture.s3.ap-northeast-2.amazonaws.com/public/clip_video_poster/4fd5b88d010e4079bc73f74786e98596/1b900ab8-0097-4463-9839-d18b52d71cc2.png)](https://slid.cc/docs/4fd5b88d010e4079bc73f74786e98596)


 - 결론 : 시각기능이 인간도 Bias가 포함되어 학습되어있으며,&nbsp;인간의 시각능력도 불완전하다.







[![(1강) Image classification 1 video poster](https://slid-capture.s3.ap-northeast-2.amazonaws.com/public/clip_video_poster/4fd5b88d010e4079bc73f74786e98596/1364e05d-b69e-45c4-9ff8-93cf0e2c4822.png)](https://slid.cc/docs/4fd5b88d010e4079bc73f74786e98596)




[![(1강) Image classification 1 video poster](https://slid-capture.s3.ap-northeast-2.amazonaws.com/public/clip_video_poster/4fd5b88d010e4079bc73f74786e98596/fdd2872e-65b2-45c9-b572-de098996df55.png)](https://slid.cc/docs/4fd5b88d010e4079bc73f74786e98596)

[![(1강) Image classification 1 video poster](https://slid-capture.s3.ap-northeast-2.amazonaws.com/public/clip_video_poster/4fd5b88d010e4079bc73f74786e98596/c2f889fc-b367-4e62-852e-ed8601fd0d3e.png)](https://slid.cc/docs/4fd5b88d010e4079bc73f74786e98596)





머신러닝과 딥러닝의 차이


 - 머신러닝 : Feature추출을 인간이 개입하여 해주고, 간단한 분류와 같은 Task 해결
 - 딥러닝 : 알아서 혼자 학습하게 됨, Feature 추출도 딥러닝학습이 해준다.




[![(1강) Image classification 1 video poster](https://slid-capture.s3.ap-northeast-2.amazonaws.com/public/clip_video_poster/4fd5b88d010e4079bc73f74786e98596/a30d5109-ee15-4f06-bc11-92362aa28ffc.png)](https://slid.cc/docs/4fd5b88d010e4079bc73f74786e98596)


딥러닝은 선입견이 별로 개입되지 않는다.


인간이 직접 하나하나 Feature추출하는 것보다.


딥러닝의 수식기반 알고리즘이 인간보다 더 좋을 수 있더라.





CVPR : 컴퓨터 비전과 패턴 인식 컨퍼런스




[![(1강) Image classification 1 video poster](https://slid-capture.s3.ap-northeast-2.amazonaws.com/public/clip_video_poster/4fd5b88d010e4079bc73f74786e98596/87ccd32d-f8d8-40a1-9758-a2c4eba5ba7b.png)](https://slid.cc/docs/4fd5b88d010e4079bc73f74786e98596)


기본적인 이미지 태스크들


현실에서 사용할만한 테크닉들




[![(1강) Image classification 1 video poster](https://slid-capture.s3.ap-northeast-2.amazonaws.com/public/clip_video_poster/4fd5b88d010e4079bc73f74786e98596/e0b75247-cb1f-424c-8ddc-7a98b93b36b1.png)](https://slid.cc/docs/4fd5b88d010e4079bc73f74786e98596)


다른 Modal data들 다루기


다양한 Visualization 툴도 다루기







[![(1강) Image classification 1 video poster](https://slid-capture.s3.ap-northeast-2.amazonaws.com/public/clip_video_poster/4fd5b88d010e4079bc73f74786e98596/7f20c7d4-93cd-4dba-ad6c-ee121d7ce506.png)](https://slid.cc/docs/4fd5b88d010e4079bc73f74786e98596)


어떤 분류에 속하는지 mapping하는 분류기







[![(1강) Image classification 1 video poster](https://slid-capture.s3.ap-northeast-2.amazonaws.com/public/clip_video_poster/4fd5b88d010e4079bc73f74786e98596/0887c84c-7905-4c10-ba50-33299e600ce1.png)](https://slid.cc/docs/4fd5b88d010e4079bc73f74786e98596)


이 세상의 모든 데이터를 가지고 있다면, 거리가 가까운 애들을 찾아주는 알고리즘의 간단하게 KNN가지고도 풀 수 있음.




## 0. Introduction

## 0-0) KNN

[![(1강) Image classification 1 video poster](https://slid-capture.s3.ap-northeast-2.amazonaws.com/public/clip_video_poster/4fd5b88d010e4079bc73f74786e98596/532f8c7e-01ac-442d-9771-379ccbf64988.png)](https://slid.cc/docs/4fd5b88d010e4079bc73f74786e98596)


KNN 설명







[![(1강) Image classification 1 video poster](https://slid-capture.s3.ap-northeast-2.amazonaws.com/public/clip_video_poster/4fd5b88d010e4079bc73f74786e98596/2df59c23-4ddd-4599-8236-04da51225e82.png)](https://slid.cc/docs/4fd5b88d010e4079bc73f74786e98596)


KNN 매우 좋아보이지만, 현실 데이터를 다 담아서 학습한다는게 말이 안됨.


 KNN 단점과 한계


 - 시간, 메모리 복잡도 측면
 - 이미지끼리 서로 유사하다, 가깝다의 정의를 인간이 해주어야 함.(쉬운 문제가 아님)




## 0-1) Single Layer

[![(1강) Image classification 1 video poster](https://slid-capture.s3.ap-northeast-2.amazonaws.com/public/clip_video_poster/4fd5b88d010e4079bc73f74786e98596/134206ce-92f2-4d8a-895d-7bf8883852b1.png)](https://slid.cc/docs/4fd5b88d010e4079bc73f74786e98596)


가장 간단한 Single Layer (Fully Connected Layer)


 - 모든 픽셀들을 각각 가중치를 곱해줘서 합해주고 활성화함수에 넣어서 분류 score를 출력


한계가 존재함.







[![(1강) Image classification 1 video poster](https://slid-capture.s3.ap-northeast-2.amazonaws.com/public/clip_video_poster/4fd5b88d010e4079bc73f74786e98596/efbf2503-e2cf-4dc9-af9c-fc97d181e313.png)](https://slid.cc/docs/4fd5b88d010e4079bc73f74786e98596)

[![(1강) Image classification 1 video poster](https://slid-capture.s3.ap-northeast-2.amazonaws.com/public/clip_video_poster/4fd5b88d010e4079bc73f74786e98596/b12f66b6-5860-4ba4-8b06-9fbe7bbd0ecd.png)](https://slid.cc/docs/4fd5b88d010e4079bc73f74786e98596)

[![(1강) Image classification 1 video poster](https://slid-capture.s3.ap-northeast-2.amazonaws.com/public/clip_video_poster/4fd5b88d010e4079bc73f74786e98596/e123ef52-170f-48b7-9956-8eb5143da0af.png)](https://slid.cc/docs/4fd5b88d010e4079bc73f74786e98596)


weight를 image에 맞추어 reshaping해서 학습을 한다고 생각해보면,


 - 평균 이미지같은 것 외에는 표현이 안됨.


 - 전체사진만을 확인하여 학습해내었기 때문에, 트레인시에 잘 보지 못했기 때문에&nbsp;
 - 테스트시에 잘린 데이터가 들어오면 성능을 발휘하지를 못함. (Sofa를 출력한다거나 하는 실수를 범함)




[![(1강) Image classification 1 video poster](https://slid-capture.s3.ap-northeast-2.amazonaws.com/public/clip_video_poster/4fd5b88d010e4079bc73f74786e98596/e9dd8b9d-c825-485e-9f47-6b3b6160ca66.png)](https://slid.cc/docs/4fd5b88d010e4079bc73f74786e98596)




## 1. CNN

[![(1강) Image classification 1 video poster](https://slid-capture.s3.ap-northeast-2.amazonaws.com/public/clip_video_poster/4fd5b88d010e4079bc73f74786e98596/07e0cd18-f386-443d-a7e3-c9cf1cba9390.png)](https://slid.cc/docs/4fd5b88d010e4079bc73f74786e98596)


위와 같은 한계들 때문에 CNN이 나옴.


FC layer 대신에 local하게 특징을 뽑아내는 구조를 사용하였음.







[![(1강) Image classification 1 video poster](https://slid-capture.s3.ap-northeast-2.amazonaws.com/public/clip_video_poster/4fd5b88d010e4079bc73f74786e98596/c7b821a6-4740-4cc7-aebb-296d1ee0c948.png)](https://slid.cc/docs/4fd5b88d010e4079bc73f74786e98596)


weight(parameter) 수가 더 적은 파라미터로 성능이 오히려 더 좋음.

[![(1강) Image classification 1 video poster](https://slid-capture.s3.ap-northeast-2.amazonaws.com/public/clip_video_poster/4fd5b88d010e4079bc73f74786e98596/672f9669-fccb-4036-ae76-a380aa24a762.png)](https://slid.cc/docs/4fd5b88d010e4079bc73f74786e98596)




[![(1강) Image classification 1 video poster](https://slid-capture.s3.ap-northeast-2.amazonaws.com/public/clip_video_poster/4fd5b88d010e4079bc73f74786e98596/e10d0a1f-c784-4c33-a9b7-7530bc957d67.png)](https://slid.cc/docs/4fd5b88d010e4079bc73f74786e98596)


CNN은 다양한 CV&nbsp; task의 기반이 됨.




[![(1강) Image classification 1 video poster](https://slid-capture.s3.ap-northeast-2.amazonaws.com/public/clip_video_poster/4fd5b88d010e4079bc73f74786e98596/4c7bef3d-903f-43c4-a67b-7f5f0fa4a3b6.png)](https://slid.cc/docs/4fd5b88d010e4079bc73f74786e98596)







[![(1강) Image classification 1 video poster](https://slid-capture.s3.ap-northeast-2.amazonaws.com/public/clip_video_poster/4fd5b88d010e4079bc73f74786e98596/16c66144-ea16-4967-92bb-7064de6cefda.png)](https://slid.cc/docs/4fd5b88d010e4079bc73f74786e98596)


CNN 아키텍쳐 기반의 image classfier들의 과정


(AlexNet, VGGNet, GoogLeNet, ResNet, Beyond ResNet)




## 2. AlexNet

[![(1강) Image classification 1 video poster](https://slid-capture.s3.ap-northeast-2.amazonaws.com/public/clip_video_poster/4fd5b88d010e4079bc73f74786e98596/5faec555-ee74-4fb6-866f-f8299c3d8da9.png)](https://slid.cc/docs/4fd5b88d010e4079bc73f74786e98596)


LeNet에서 기본적인 구조를 따왔지만, 한단계 더 나아간 point를 가진 AlexNet




[![(1강) Image classification 1 video poster](https://slid-capture.s3.ap-northeast-2.amazonaws.com/public/clip_video_poster/4fd5b88d010e4079bc73f74786e98596/b3b1e7cd-1389-43c9-ae07-dfe3d791edf8.png)](https://slid.cc/docs/4fd5b88d010e4079bc73f74786e98596)


모델사이즈도 더 크고, 학습데이터도 엄청 크고, 활성화함수 드롭아웃 등도 추가됨.




[![(1강) Image classification 1 video poster](https://slid-capture.s3.ap-northeast-2.amazonaws.com/public/clip_video_poster/4fd5b88d010e4079bc73f74786e98596/479d279f-395b-4a7f-b76c-48bfd114063f.png)](https://slid.cc/docs/4fd5b88d010e4079bc73f74786e98596)


이 때 당시에는 GPU가 모자라서 GPU 두개에 따로 연산하고, 일부에서만 서로 ineraction이 일어나게끔 모델이 설계되었음.







[![(1강) Image classification 1 video poster](https://slid-capture.s3.ap-northeast-2.amazonaws.com/public/clip_video_poster/4fd5b88d010e4079bc73f74786e98596/139a4ead-d0a1-4a66-a3d4-940d69f957a1.png)](https://slid.cc/docs/4fd5b88d010e4079bc73f74786e98596)





x

[![(1강) Image classification 1 video poster](https://slid-capture.s3.ap-northeast-2.amazonaws.com/public/clip_video_poster/4fd5b88d010e4079bc73f74786e98596/7a9c3274-629d-418f-8e0b-de2fb7b789cf.png)](https://slid.cc/docs/4fd5b88d010e4079bc73f74786e98596)


tensor를 받아서 연산할 수 없으니까 차원을 matrix로




[![(1강) Image classification 1 video poster](https://slid-capture.s3.ap-northeast-2.amazonaws.com/public/clip_video_poster/4fd5b88d010e4079bc73f74786e98596/aa0ac725-0f3f-4030-b573-77efdbde4b80.png)](https://slid.cc/docs/4fd5b88d010e4079bc73f74786e98596)


Flattening 아니면 에버리지 풀링으로.




[![(1강) Image classification 1 video poster](https://slid-capture.s3.ap-northeast-2.amazonaws.com/public/clip_video_poster/4fd5b88d010e4079bc73f74786e98596/fe7ef087-67ff-4915-ba4f-d77e24111cb1.png)](https://slid.cc/docs/4fd5b88d010e4079bc73f74786e98596)







[![(1강) Image classification 1 video poster](https://slid-capture.s3.ap-northeast-2.amazonaws.com/public/clip_video_poster/4fd5b88d010e4079bc73f74786e98596/3dc05e56-7197-46c4-827e-ad78f4b33016.png)](https://slid.cc/docs/4fd5b88d010e4079bc73f74786e98596)


LRN은 명암 구조를 설명한다고 이해하면 됨.


요즘은 BN이 흔하게&nbsp; 사용되고 있음







[![(1강) Image classification 1 video poster](https://slid-capture.s3.ap-northeast-2.amazonaws.com/public/clip_video_poster/4fd5b88d010e4079bc73f74786e98596/13287993-5111-47a1-8639-6ab15388dea3.png)](https://slid.cc/docs/4fd5b88d010e4079bc73f74786e98596)




![](https://slid-capture.s3.ap-northeast-2.amazonaws.com/public/capture_images/4fd5b88d010e4079bc73f74786e98596/4cdbaf86-f26f-4b12-99ea-4966aa504654.png "(1강) Image classification 1 image")


필터사이즈를 크게 썼었지만, 최신네트워크 구조에선 작게 쓰고 있음.





CNN구조에서 Receptive field란 위처럼 Layer의 한 element가 나오기까지 연관이 있었던 맨 앞 층 input의 영역을 의미한다.


 - 예를 들어 여기선 3X3 conv filter를 썼으니까 표시된 3X3 영역이 영향을 주었음, 이런식으로 여러 layer를 거치면서 결과값 하나에 연관된 input의 부분이 receptive field이다.


 - 이런 식이면 오른쪽 그림처럼 층이 여러개 쌓일 수록 맨 앞단의 참고했던, 연관이 있었던 receptive 영역은 점점 커지게 된다..
 - 위에서 언급했듯&nbsp;receptive fields는 연관되어있는 input space에서의 영역을 의미하기 때문에 매우 방대한 크기일 것이다.

[![(1강) Image classification 1 video poster](https://slid-capture.s3.ap-northeast-2.amazonaws.com/public/clip_video_poster/4fd5b88d010e4079bc73f74786e98596/9df0e80e-b9c1-4771-a30e-85b549d38143.png)](https://slid.cc/docs/4fd5b88d010e4079bc73f74786e98596)




[![(1강) Image classification 1 video poster](https://slid-capture.s3.ap-northeast-2.amazonaws.com/public/clip_video_poster/4fd5b88d010e4079bc73f74786e98596/516a8182-ed25-4a5e-b008-dae671c2cdaf.png)](https://slid.cc/docs/4fd5b88d010e4079bc73f74786e98596)


두 layer 사이에서 receptive field 영역 크기를 계산하는 공식은 (P+K-1)X(P+K-1) 와 같다.




## 2. VGGNet

[![(1강) Image classification 1 video poster](https://slid-capture.s3.ap-northeast-2.amazonaws.com/public/clip_video_poster/4fd5b88d010e4079bc73f74786e98596/03046a6d-5bfb-4815-800a-5e6e28e9bdc7.png)](https://slid.cc/docs/4fd5b88d010e4079bc73f74786e98596)


VGGNet의 기본적인 특징


1. AlexNet보다 층이 깊다.


 - VGG16 (16 layers) , VGG19 (19 layers)


2. 더 간단한 Architecture를 이용했음.


 - LRN (local Normalization)을 사용하지 않았다.
 - AlexNet에서는 11X11이었지만, 3X3 conv filter, 2X2의 pooling만을 사용했다.




[![(1강) Image classification 1 video poster](https://slid-capture.s3.ap-northeast-2.amazonaws.com/public/clip_video_poster/4fd5b88d010e4079bc73f74786e98596/0028664b-1681-4c8c-8b77-fe71369759df.png)](https://slid.cc/docs/4fd5b88d010e4079bc73f74786e98596)







[![(1강) Image classification 1 video poster](https://slid-capture.s3.ap-northeast-2.amazonaws.com/public/clip_video_poster/4fd5b88d010e4079bc73f74786e98596/0cef5025-eb52-41a2-a1aa-6675f5a3bf8c.png)](https://slid.cc/docs/4fd5b88d010e4079bc73f74786e98596)


이런 간단한 구조이지만, 매우 좋은 성능을 발휘하였음


미리 학습된 중간 feature들을 다른 task에 쓸 수 있을 정도로 일반화가 잘 되어있음. (Fine-tuning을 통해서 사용할 때)




[![(1강) Image classification 1 video poster](https://slid-capture.s3.ap-northeast-2.amazonaws.com/public/clip_video_poster/4fd5b88d010e4079bc73f74786e98596/4f8c5d93-4cb2-4c5f-9a11-ef4c2111ea89.png)](https://slid.cc/docs/4fd5b88d010e4079bc73f74786e98596)


AlexNet과 input 형태는 똑같음. (224X224 image 3 channel)


training data에서 평균 RGB값을 각 채널에서 빼주면서 input으로 넣어줌. (Normalization 느낌)

[![(1강) Image classification 1 video poster](https://slid-capture.s3.ap-northeast-2.amazonaws.com/public/clip_video_poster/4fd5b88d010e4079bc73f74786e98596/280dcbd4-bf52-465b-8859-7e384b1c4685.png)](https://slid.cc/docs/4fd5b88d010e4079bc73f74786e98596)


stack을 (layer를) 많이 쌓았기 때문에 conv filter와 pooling size가 작더라도 더 큰 receptive filed를 input으로부터 반영할 수 있다.


즉, 더 작은 parameter만을 가지고도 input에서 더 많은 부분들을 고려해서 학습을 하게 된다는 것이다.







[![(1강) Image classification 1 video poster](https://slid-capture.s3.ap-northeast-2.amazonaws.com/public/clip_video_poster/4fd5b88d010e4079bc73f74786e98596/a1b91e62-3295-4040-8d33-7ac0694df9d2.png)](https://slid.cc/docs/4fd5b88d010e4079bc73f74786e98596)


맨마지막 layer는 3 FC layer로 되어있다.


 AlexNet과 마찬가지로 Relu를 사용하여 non-linearity를 학습하게 된다.


Local Response Normalization은 사용하지 않았다.



