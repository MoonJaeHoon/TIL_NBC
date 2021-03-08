# 캠퍼 질문

## 1. BERT의 Transfer Learning관련 궁금증이 생겨 질문드립니다.

![img](https://cphinf.pstatic.net/mooc/20210219_241/1613707839216vRe9M_PNG/mceclip0.png)

교수님께서 설명하시길, sentence pair인 경우 [SEP] 토큰으로 문장을 구별지은 후, [CLS] 토큰을 output layer의 입력으로 준다. 라고 하셨고

single sentence의 경우는 [CLS] 토큰을 output layer의 입력으로 준다고 하셨습니다.



**제가 궁금한 점은 두 가지입니다.**

**1) 두 개의 task가 '[SEP] 토큰으로 문장을 구별짓는다' 를 제외하면 같은 방식으로 수행이 되는가**

**2) Sentence pair의 경우 [SEP] 토큰을 기준으로 나뉜 뒷 부분의 문장의 처리를 어떻게 하는가(모두 [CLS] token에 정보가 저장 되는지?)**



> 1. 네, 실질적으로 바뀌는 부분은 input밖에 없습니다. 다시 말해, 정확히 동일한 모델로 2가지의 task 모두를 수행할 수 있습니다. 보시는 것처럼 [SEP] token 등 special token 등을 이용해 입력 sequence에 구분을 두어 task를 수행하는 식으로 pre-train 모델을 많이 활용하고 있는 것 같습니다.

> 2. 네, [CLS] token은 pre-training task 중 next sentence prediction을 수행할 때, output layer의 입력으로 사용되어 입력 sequence의 context를 담은 token으로 활용되고 있습니다. 때문에 문장 단위의 task를 수행하는 경우 많은 모델들이 output layer의 입력을 [CLS] token의 output hidden state로 사용합니다.
>    사실 꼭 문장 단위의 정보를 전달하기 위해 CLS token을 써야하는 것은 아닙니다만, BERT에서 처음 해당 방식이 제안되어서 동일한 방식으로 활용해오고 있는 것 같습니다. 최근에는 이 transformer block 다음 output layer 부분을 단순히 linear layer 하나만 두지 않고 attention을 활용하기도 하는 것 같습니다. 
>
> (참고: Improving Commonsense Question Answering by Graph-based Iterative Retrieval over Multiple Knowledge Sources, https://www.aclweb.org/anthology/2020.coling-main.232.pdf, 5페이지)





## 2. self-supervised learning 관련 질문드립니다.

self-supervised learning 과 pretrained model 개념이 와닿지 않아 질문드립니다.

제가 이해하는 self-supervised learning 은 

1. pretext task (연구자가 직접 정의한 task) 를 미리 정의한 후

2. label 이 없는 데이터셋을 활용하여 모델을 학습시키는데, 이때, supervision(지도) 를 준다.

3. 2에서 학습된 모델을 downstream task(target task)로 가져와 transfer learning 을 수행한다

4. 결론적으론 처음엔 label 이 없는 상태에서 직접 supervision을 만들어 학습후 transfer learning 단계에서는 label 이 있는 supervised learning을 수행한다.

입니다.



이때, 제가 질문드리고 싶은 사항은 다음과 같습니다.

1. supervision을 주는 상황은 어떤 방식으로 진행될까요? semi-supervised learning 처럼 아주 일부분만 label를 주는걸까요?

2. self-supervised pretrained-model 이라는 것은 위의 과정 처럼 self-supervised 학습을 통해 얻은 weight를 가져와 우리가 원하는 target task로 적용하는 것을 의미하나요?

3. supervision을 준다는 점에서 supervised learning으로 보이는데 큰 분류에서 바라보면 supervised learning을 의미하는 걸까요?

정확한 개념이 서지 않아 질문이 길어졌습니다.



> self-supervised learning은 자체적으로 label을 만들어 모델을 학습하는 방법이고. 말씀해주신 일련의 과정은 transfer learning에 대한 설명에 가까운 것 같습니다. 모델을 꼭 self-supervised learning 방법으로 학습시킬 필요는 없고 그저 대용량의 dataset에 대해 학습한 model의 knowledge를 작은 dataset을 학습하는 데에 활용하는 방법이라고 생각해주시면 좋을 것 같습니다. 아래의 그림을 참고해주시면 될 것 같고 질문 사항에 대해서 하나씩 말씀드리면요.
>
> 1. supervision을 주는 방식은 여러가지가 있을 수 있겠지만, BERT에서는 단어 일부를 가리고(mask) 원래 이 단어가 무엇이었을지 맞추는 cloze task에서 영감을 받았다고 이야기 합니다. (참고: https://en.wikipedia.org/wiki/Cloze_test) 이 방식으로 학습을 하는 경우 강의에서 보신 것처럼 label을 자체적으로 만들어 self-supervised learning이 가능합니다. 이 경우 semi-supervised learning과 달리 모든 example에 대해서 label이 존재합니다. label을 만드는 구체적인 방법은 강의에서 masked language model과 next sentence prediction을 설명하는 부분에서 찾아보실 수 있습니다. 모델은 항상 2개의 문장을 받고 2문장의 관계 그리고 15%의 확률에 걸린 token들에 대해서 classification을 수행하게 됩니다.
>
> 2. 네 정확히는 self-supervised learning을 통해 학습한 pre-trained model을 원하는 target task에 활용한다. 이 정도의 문장으로 정리할 수 있을 것 같습니다. target task에 해당 모델을 가져와서 학습을 할 때에는 pre-train때 보다 훨씬 더 적은 양의 data를 가지고 더 적은 step을 학습하기 때문에 미세조정(fine-tuning)이라는 이름으로 target task에 적용한다고 이야기하는 것 같습니다.
>
> 3. 네 말씀하신 내용이 맞습니다. 큰 범위에서 label을 활용해 학습 방법은 모두 supervised learning에 해당하고 하위 분류로 self-supervised learning, semi-supervised learning 등이 있는 것 같습니다. ( self-supervised learning은 과거 unsupervised learning의 용어 대체로 보시는게 더 좋을 것 같습니다 - https://www.facebook.com/yann.lecun/posts/10155934004262143  )

![img](https://cphinf.pstatic.net/mooc/20210219_21/1613705236479OyDqy_PNG/mceclip0.png)







## 3. BERT pretrain 학습시 질문있습니다.

[![img](https://cphinf.pstatic.net/mooc/20210219_101/1613700113190nQbCo_PNG/image.PNG)](https://cphinf.pstatic.net/mooc/20210219_101/1613700113190nQbCo_PNG/image.PNG)



여기서 교수님이 100개의 단어중에 15%인 15개 단어에 대해 마스크를 수행할 때,
15개 단어 중 80프로인 12개 단어는 마스크를 수행하고,
10프로인 1.5개의 단어는 랜덤워드를 수행하라고 말하셨는데

제가 저 문장을 읽으면서 느낀 것은
15개 단어가 등장하는 문장의 80%는 마스크를 수행하고 10%는 랜덤 워드, 10%는 단어 그대로 넣어주는 것으로 보입니다. 혹시 어떤게 맞는 방법인가요??



> 네 교수님이 말씀하신게 맞습니다. 100개의 단어가 있으면 그 중 15%의 단어가 해당됩니다. 말씀하신대로 100개 중 15개의 단어라고 하면 15개 중 80%가 mask token으로 바뀌고 10%는 random word, 10%는 원래 단어 그대로 남게 됩니다. 슬라이드의 설명이 조금 부족했네요