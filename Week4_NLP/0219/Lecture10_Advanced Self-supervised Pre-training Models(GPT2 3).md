# Advanced Self-supervised Pre-training Models

GPT-1과 BERT 이후에 나온 자기지도 사전학습 모델을 알아보자.

## GPT-2

**`GPT-2`**는 GPT-1에 비해 다음과 같은 점이 발전되었다.

- Layer를 훨씬 더 많이 쌓았다. 다만 학습 방식은 Language Modeling으로 동일하다.
- 학습 데이터가 40GB로 늘어났다. 게다가 데이터의 퀄리티도 좀 더 신경써서 준비했다고 한다.
- Language Model이 메인 task 이외의 down-stream task들을 zero-shot setting, 즉 파라미터나 아키텍쳐 조작을 전혀 하지 않고도 수행가능하다는 것을 입증했다.

모티베이션이 된 논문은 다음과 같다.

[The Natural Language Decathlon : Multitask Learning as Question Answering](https://arxiv.org/pdf/1806.08730.pdf)

- **모든 종류의 자연어 처리 task들이 Question-Answering task로서 처리될 수 있다**는 점을 시사하였다.

### 데이터셋

- Reddit외부링크를 포함한 모든 웹 텍스트를 수집했다.총 45M개의 링크를 사용했다.사람이 직접 큐레이팅/필터링한 웹페이지만 수집하였다.3개 이상의 추천을 받은 게시글들만 수집하였다.
- 8M 크기의 위키피디아 문서
- 링크로부터 content를 추출하기 위하여 dragnet과 newspaper를 사용했다고 한다.

### 전처리

**`Byte Pair Encoding(BPE)`**를 사용하였다.

### (GPT-1 대비) 구조상의 변화

![gpt-2](https://blogik.netlify.app/static/3ac1124d41c7ce6a6de54211ce0ef74c/80b2d/gpt-2.png)

- Layer Normalization이 각 sub-block의 입력쪽으로 옮겨졌다(pre-activation residual network와 비슷한 형태).
- 마지막 self-attention block에서 Layer Normalization이 하나 추가되었다.
- 각 Layer를 random initialization할 때, Layer의 깊이에 비례해서 값이 더 작아지도록 초기화하였다. 따라서 위쪽에 있는 Layer가 하는 역할이 좀 더 축소되도록 하였다.

### Zero-Shot Setting

**`Zero-Shot Setting`**이란, **데이터가 없이도 해당 task의 수행을 할 수 있음**을 일컫는 용어이다.

- Question AnsweringTask별 학습을 시키지 않고 pre-training만 시킨 GPT-2에 질문(Question)을 준 뒤, 다음에 나올 문장을 예측하라고 했더니, F_1*F*1 스코어가 55가 나왔다고 한다. 잘 학습된 BERT 모델이 89의 F_1*F*1스코어를 기록하는 것을 생각해보면, 학습도 시키지 않은 것 치고는 준수한 성능이다.
- SummarizationCNN과 일간 메일 데이터셋에서 TL;DR(Too Long, Didn't Read) 이라는 어구를 기준으로 요약이 있었는데, 이 때문에 TL;DR을 GPT-3에 입력으로 주면 요약을 해 준다.
- TranslationSummarization과 같은 방식으로, [In French, In Korean, ...]이라는 어구가 있으면 번역까지 수행해준다고 한다.

## GPT-3

가장 최근에 나온 GPT 모델로, 모델의 구조에 변화가 있었다기보다는, **이전과 비교할 수 없을 정도의 attention block을 쌓아 파라미터수를 어마어마하게 많이 늘렸다(150B)**. 또, **배치사이즈도 3.2M정도가 되도록 최대한 키우자 더 좋은 성능**을 보였다고 한다.

![gpt-3-few-shot](https://blogik.netlify.app/static/d51200945978aec912e03f6011da3c6e/21521/gpt-3-few-shot.png)

GPT-2에 비해 눈에 띄는 특징은 다음과 같다.

- GPT-2에서는 '가능성'정도로만 보였던 `Zero-shot setting`이 놀라운 수준으로 발전하였다.학습에서 전혀 활용하지 않았던 텍스트를 translation 했을때도 정상적으로 기능한다.
- 하고자 하는 task(예를 들자면 번역)를 주고, 예시를 주면, 자연어 생성 task로 인식하여 정확도를 평가하고 스스로 학습한다. 이를 **`One-shot`**이라고 한다. 데이터를 단 한 쌍(예시)만 주었다는 말이다.신기한 점은, 모델 자체의 파라미터를 변경시켜가며 학습한 것이 아니라, 데이터를 input 텍스트의 일부로서 제시했는데도 task를 수행했다는 것이다!
- 동일한 맥락으로, 몇 개 안되는 예시 데이터를 주고 task를 수행하도록 하는 Few-shot이 가능해졌다.

![gpt-3-few-shot-performance](https://blogik.netlify.app/static/60116e0b144deab5846ef610322313da/2bef9/gpt-3-few-shot-performance.png)

연구 결과에 따르면, 모델 사이즈를 키우면 키울수록, **`Zero/One/Few shot`**의 성능이 계속해서 오른다고 한다.

## ALBERT

**`A Lite BERT(ALBERT)`**는 기존의 BERT를 경량화 시킨 모델이다. GPT와 같이 모델이 굉장히 거대해지고 리소스와 연산량이 많아지는 형태와는 달리, 오히려 복잡했던 BERT를 개선하는 데에 집중했다. 모델 사이즈를 줄이고, 학습시간과 리소스를 줄이면서도 성능은 크게 떨어뜨리지 않는 경량화 형태의 Pre-trained Model이다.

- BERT에서 큰 버젼과 작은 버젼이 있듯이, ALBERT에도 모델 파라미터 사이즈에 따라 좀 더 큰 모델과 작은 모델이 있다. 물론 더 큰 모델을 사용할 때 좀 더 좋은 성능을 낸다.

이해하기 쉽게 잘 정리된 글이 있으니 아래의 내용들이 이해가 가지 않는다면 이 링크를 참조하자.

[[ALBERT 논문 Review\] ALBERT: A LITE BERT FOR SELF-SUPERVISED LEARNING OF LANGUAGE REPRESENTATIONS](https://y-rok.github.io/nlp/2019/10/23/albert.html)

### Factorized Embedding Parameterization

![factorized-embedding-parameterization](https://blogik.netlify.app/static/d1f32130e74a44018411d7a41f19984d/2bef9/factorized-embedding-parameterization.png)

기존의 BERT에서 Embedding vector 사이즈 E*E*는 hidden vector size H*H*와 항상 같아야했다. 여러 Attention Block을 쌓기 때문에, 같은 크기로 들어가고 나가야 다음 블록에 동일한 형태로 전달될 수 있다.

문제는, 단어간의 관계를 인코딩하여 저장해야하므로 많은 정보가 들어가는 dependent 벡터 H*H*의 크기를 맞추기 위해, **단어간의 관계를 생각하지 않아도 되는 independent 벡터 E\*E\*가 필요 이상으로 커진다**는 것이다.

![albert-fep2](https://blogik.netlify.app/static/dc55109428294d55d3be60ea96110ffa/2bef9/albert-fep2.png)

위 이미지를 보면, BERT에서는 원래 4x1 사이즈의 임베딩 벡터가 H에 맞춰주기 위하여 4x4로 늘어나는 것을 볼 수 있다.

이를 해결하기위해 ALBERT는 **Embedding Matrix를 위의 이미지처럼 두 Matrix의 곱으로 쪼갠다**. 가령 H(=4)에 맞추기 위하여 4x4였던 행렬을, 4x2 크기의 행렬로 두고 추가적으로 Layer를 하나 더 두어, Word별로 구성되는 2차원 벡터를 4차원으로 선형변환(W)시켜주도록 한다. 이 때 W는 H의 크기로 변환될 수 있도록 적당한 크기를 가지면 된다.

- **`row-rank matrix factorization`**이라고 하는 기법이다.
- 위의 예제에서는 잘 와닿지 않지만, H의 크기가 100이고 쪼갠 Matrix의 column length는 10이라고 생각해보자. 파라미터 수가 확 줄어듦이 체감될 것이다.

### Cross-layer Parameter Sharing

Self-attention block들이 가지는 학습 파라미터들에는 무엇들이 있을까?

- 임베딩된 input 벡터가 Query, Key, Value 각각의 역할을 하도록 변형시켜주는 [W^Q,W^K,W^V][*W**Q*,*W**K*,*W**V*]multi head라면 행렬 세트가 총 head개가 된다.
- Z^t*Z**t*들을 concat한 후 다시 원래의 Hidden State Vector 크기의 Z*Z*로 줄여주기 위한 W^O*W**O*

물론, 각각의 Self Attention block마다 이 파라미터값은 모두 다를것이다.

그런데 ALBERT는 서로 다른 Layer, 즉 **서로 다른 Self-Attention Block에 존재하는 파라미터들을 서로 공유**하는 방법을 제시한다. 이를 **`Cross-layer Parameter Sharing`**이라고 한다.

![albert-sharing](https://blogik.netlify.app/static/a3bac524d238aad508f7ecf09fa6c8ab/2bef9/albert-sharing.png)

- **`Shared-FFN`** : Layer 간에 feed-forward network의 파라미터만 공유한다.
- **`Shared-attention`** : Layer 간에 attention 파라미터들만 공유한다.
- **`All-shared`** : 둘 다 공유한다.

이처럼 파라미터를 공유하였을 때, **파라미터의 개수는 크게 줄었음에도 불구하고 모델의 성능은 그다지 크게 떨어지지 않았음을 입증**하였다.

### Sentence Order Prediction

BERT 이후의 후속연구에서, BERT 모델이 기존에 pretraining하던 `Next Sentence Prediction task`는 사실 너무 쉬워서 그다지 실효성이 없는 것으로 드러났다.

- 두 문장의 출처가 다르다면, 사실상 전혀 다른 내용일 확률이 높다.
- 따라서 그냥 비슷한 단어나 문맥이 많이 등장했는가 정도로 선후관계를 파악하게 된다.
- 이는 선후 관계보다는 topic prediction에 가깝다.

ALBERT에서는 해당 task의 pretraining을 빼고 좀 더 유의미한 task들을 집어넣어, 모델의 성능을 확장했다.

- 두 독립적인 문장을 가져와 선후관계를 파악하는 것이 아니라, 항상 연속적인 두 문장을 가져온다.
- 그 문장을 원래의 순서대로 concat했을 때 정방향으로 예측하고, 역순으로 concat했을 때 역방향으로 예측하도록 학습시킨다.(이진분류)
- 이를 `negative sample`이라고 하는데, 인접 문장이므로 순서와 관계없이 비슷한 단어가 당연히 많이 등장한다.따라서 정말로 논리적인 흐름을 주의깊게 파악해야 task를 풀 수 있는 pretraining 형태가 되었다.

![albert-sop](https://blogik.netlify.app/static/bbb2c2706f67613d7b2146849ebe59b7/2bef9/albert-sop.png)

논문에 첨부된 위의 실험결과를 보면, Next Sentence Prediction(NSP)를 사용했을때는 아예 사용하지 않았을 때와 별 차이가 없거나 오히려 성능이 떨어지기까지 한다. 이에 비해 **`Sentence Order Prediction(SOP)`**는 좀 더 좋은 개선된 성능을 보여주고 있다.

## ELECTRA

2020년에 발표된 논문으로, GPT의 standard한 LM이나, BERT의 Masked LM task에서 나아가, **GAN 형태의 모델링**을 제시하고 있다.

MLM(Masked language Modeling)을 통해 마스킹된 단어를 복원해주는 모델-**`Generator`**를 하나 두고, 또 Generator가 복원한 단어들을 받아 이 단어가 원본인지 또는 generator에 의해 복원된 단어인지를 예측하는 모델-**`Discriminator`**를 둔다.

- generator는 BERT의 형태로 볼 수 있다.
- 이 때 Generator와 Discriminator이라는 구조를 Generative Adversarial Network(GAN)형태로 볼 수 있다.

이렇게 모델 학습을 진행할 경우에, Pre-train된 모델로서 사용할 수 있는 부분이 Generator와 Discriminator 두 부분이 된다. 이 중 **`ELECTRA`**는 Discriminator를 가져다가 downstream task에 맞게 fine-tuning하여 사용하는 방식이다.

![electra-performance](https://blogik.netlify.app/static/30a0c7ee8b543008c320d70db20172b3/2bef9/electra-performance.png)

ELECTRA의 논문에 따르면, **대부분의 BERT 모델보다 동일한 학습량 대비 성능이 더 좋다**고 한다.

## Light-weight Models

이외에도 최근의 모델 연구동향은 경량화 기술에 주목하고 있다. GPT-3같은 대형 모델의 성능은 놀랍지만, 연구나 활용을 하기에는 오히려 접근이 힘들다는 문제가 있다.

기존의 정확도를 유지하면서도 파라미터수나 레이어 수를 줄임으로써 모델의 크기와 학습속도를 빠르게 하려는 노력들이 이어지는 중이다. 이를 통해 클라우드나 서버를 통한 AI 적용이 아니라 휴대폰이나 IOT기기에서도 딥러닝이 가능하게 할 수 있다.

### DistillBERT

[Huggingface](https://huggingface.co/)에서 2019년도 발표한 모델로, Teacher 모델과 Student 모델로 이루어져있다. 큰 사이즈의 Teacher 모델이 기존의 방식으로 학습을 수행한 후, Student 모델은 더 적은 파라미터로 Teacher 모델의 수행방식을 모사하여 비슷한 성능을 낸다.

- Teacher 모델이 예측한 결과를 Student 모델이 softmax에 주는 ground-truth로 삼아 학습한다.

### TinyBERT

2020년 발표된 `TinyBERT` 역시 BERT를 경량화시킨 모델이지만, 좀 더 발전된 형태이다.

- DistillBERT과 마찬가지로 target distribution을 모사한 뒤, Student 모델에서 이를 ground-truth로 삼고 학습한다.
- 추가적으로, Student 모델이 Teacher 모델의 Attention Matrix와 Hidden state Vector까지도 유사하도록 모사한다.최종 결과물뿐만 아니라 **중간 결과물까지도 비슷하도록 모사**한다고 볼 수 있다.MSE-Loss를 사용한다.

이 때, Student 모델의 벡터 크기가 일반적으로 Teacher 모델의 벡터보다 작기 때문에 이를 욱여넣는 과정에서 전정보의 손실이 발생할 수 있다.

이를 방지하고자 TinyBERT는 벡터간의 크기 변환 과정에 Fully Connected Layer를 두어, 기존의 정보를 어느정도 유지하면서 더 적은 벡터로 이를 모사할 수 있게 된다.

## Fusing Knowledge Graph into Language Model

기존의 Pritrained 모델과 지식(knowledge) 그래프를 조합한 최신 연구동향이다.

BERT가 2018년도에 등장한 이후, BERT가 정말로 언어적인 이해를 제대로 하고 있는 것인지 분석하는 연구들이 이어지면서, BERT가 데이터셋에 나와있지 않은 종류의 문장들은 잘 이해하지 못한다는것이 드러났다.

주어진 문장에서 나타나는 지식 뿐만 아니라, 외부 지식(또는 상식)도 자연어 처리에서 중요한데, 이를 연구하는 분야가 **`Knowledge Graph`** 분야이다.

- 개체간의 관계, 개념등을 잘 정의하고, 정형화해서 만들어 둔 것을 Knowledge Graph라고 한다.

Knowledge graph를 어떻게 BERT등에 적용시킬 수 있을까에 대한 연구 동향으로 ERNIE, KagNET등이 있다.



































---

## GPT-3는 따로 어떠한 TASK를 위한 학습을 하지 않아도 성능이 뛰어나다.

프랑스어 문장 : 프1, 프2, 프3, 프4

영어문장 : 영1, 영2, 영3, 영4, 영5



프1, 프2, 프3, 프4 , Translate Token(프->영), 영1, 영2, 영3, 영4, 영5



위와 같이 문장을 하나 생성하여 학습을 시킵니다.

번역을 의미하는 토큰을 넣어주고 이런식으로 Pre-trained해놓으면, Fine-tuning시에 입력으로 프랑스어를 넣어 영어로 번역하는 TASK를 수행하려고 할 때 자동으로 높은 성능을 가질 수가 있습니다.