# Self-supervised Pre-training Models

## Recent Trends

- Transformer 및 self-attention 모델은 현재 자연어처리 뿐만 아니라 다른 분야에까지 좋은 성능을 내고있다.
- 기본적으로 Transformer의 Attention구조가 베이스가 되고있으며, Transformer 논문에 제시되었던 6개 블록만을 사용하지 않고 attention블록을 더 많이 쌓아서 만드는 추세다.
- 최근에는 Stack을 깊게 쌓은 형태의 transformer 모델들이 `전이 학습` 등을 통하여 성능을 획기적으로 향상시킨 사례가 나오고 있다.`BERT`, `GPT-3`, `XLNet`, `ALBERT`, `RoBERTa`, `Reformer`, `T5`, `ELECTRA`...
- 추천시스템, 신약개발, 컴퓨터비전 등의 기술로도 많이 사용되어 영역을 넓혀가고 있다.
- 다만 한 단계에 한 단어를 생성하는 `greedy decoding` 형태를 아직 벗어나지 못하고 있는것도 사실이다.
  - 항상 <SOS> 토큰을 시작으로 한다는 한계를 벗어나지 못 했다.

## Language model이란?

랭귀지 모델이란, 입력받은 단어의 다음 단어를 예측해주는 모델이다. 예를 들어 검색을 위해 `deep`을 타이핑했다면 추천검색어로 `deep learning`이 보여지는 것이다.

여기서 가장 큰 강점은 라벨이 없는 데이터로도 학습이 가능하다는(**비지도 학습**이라는) 점이다. 단순히 세상에 존재하는 대량의 문서나, 텍스트 등을 긁어와서 학습시키기만 하면 된다. 



이제부터 설명할 GPT는 랭귀지 모델이다.



# 1. GPT-1

일론머스크가 참여한 비영리 재단 Open AI에서 개발한 자연어 처리 모델이다.

하나의 task 뿐만 아니라 **자연어 처리와 관련된 여러 task를 모두 커버할 수 있다**는 것이 특징이다.

### 모델 구조와 작동 과정

![image-20210315055309507](Lecture9_Self-supervised Pre-training Models(GPT1_BERT).assets/image-20210315055309507.png)

- 입력 출력 시퀀스가 별도로 있지 않고, **대량의 웹 데이터**로부터 추출한 문장을 토대로 기존의 **Language Modeling Task 방식**으로 12개의 self-attention 블록이 학습되는 형식이다.

- 그러나 단순히 Lauguage Modeling 뿐만 아니라 다른 Task도 다룰 수 있게 하기 위해 새로운 Task Clasiffier 프레임워크를 제안하고 있다.
  - 한 Text에 대하여 Classification Task를 수행하면서도, 두 문장을 하나의 입력으로 주어서 다른 Task까지 수행할 수 있도록 하였다.
  - 두 문장을 넣을 때 그 사이에 delimiter를 추가하여 Special Token으로서 문장을 구분지어줬다.
- 즉, 다음 단어를 예측하는 Text precision과 Task Classifier를 동시에 수행한다.

- 문장을 넣을 때 기존에 문장 뒤에 넣어주던 `<EOS>` 토큰과 조금 다른 `<Extract>` 토큰을 넣어 인코딩한다. 인코딩 후 Extract 토큰에 해당하는 인코더의 output을 디코더에 input으로 넣어주어 (linear transformation을 거쳐) task를 위한 정보를 파악한다.
  - 이렇게 파악하게 되는 Task의 정보로는 Classification(감정분석)이나 Entailment(논리), Similarity(유사도) 등이 있다.

- 이들 중 내포(Entailment)관계라 함은 문장 A(전제,premise)가 참이면 B(가설,hypothesis)도 참인 경우를 말한다.
  - 이 경우 두 문장을 하나의 Sequence로 만들되, 문장 사이에는 Delimeter(구분자)를 넣음으로써 분리하여 input으로 삼는다.

- Extract 토큰은 처음에는 사실 단순히 문장 마지막에 추가한 토큰이었지만, Self-attention 학습과정 중에 query로 사용되어 학습된다. 결과적으로는 **Task에 필요한 정보들을 입력문장으로부터 적절하게 취합/추출할 수 있게 하는 토큰**이 된다.
  - 예를 들어, Entailment Task의 경우에는 `John은 어제 결혼했다.`와 `누군가는 어제 결혼 했다.`의 문장이 있으면 두 문장 사이에 delimiter 토큰을 넣고 하나의 sequence로 만든후 Extract 토큰을 추출하는 식이다.

### Transfer Learning

대량의 일반적인 task 데이터와 소량의 특정 task 데이터가 있는 경우, 통합적으로 학습한 GPT-1 모델을 **`전이학습(Transfer Learning)`**형태로 활용한다.

위의 아키텍쳐 이미지에서, 기존의 학습모델은 그대로 두고 출력 부분의 layer만 원하는 Main Task 분류를 위한 레이어(Task Classifier)로 수정함으로써 새로운 task에 대한 학습을 시켜줄 수 있다. 이 때 새로 들어오는 레이어는 random intialization을 거친 값으로, 빠르게 학습되어야하는 초기화된 파라미터이다. 따라서 **Pretrained 모델 부분의 Learning rate는 크게 줄이고 Main task를 위해 추가된 레이어와 함께 학습시킴**으로써, 빠르게 다른 Task에 적용시킬 수 있게 된다.



### Experimental Results

Pre-trained 된 모델을 사용하는 것이 상대적으로 소량의 labeling된 Data로 특정 Task를 수행하기 위해 modeling 된 모델들보다 정확도가 훨씬 높다는 것을 알 수 있습니다.

![image-20210315064256429](Lecture9_Self-supervised Pre-training Models(GPT1_BERT).assets/image-20210315064256429.png)



# 2. BERT (Bidectional Trasnformer)

GPT와 아이디어 자체는 동일하지만, 사전 학습시 학습시켜주는 방법에 대한 아이디어가 다르다.

현재까지도 가장 널리 쓰이는 pretraining 모델로, GPT-1과 마찬가지로 **Language Modeling 방식으로 학습**시켰습니다.

기존에 다음 단어를 예측하는 등의  LSTM을 기반으로 한 인코더 `ELMo`가 있었으나, LSTM 인코더가 Transformer로 대체되면서 BERT가 가장 많이 쓰인다.

![image-20210315064936702](Lecture9_Self-supervised Pre-training Models(GPT1_BERT).assets/image-20210315064936702.png)



## Masked Language Model (MLM)

<img src="Lecture9_Self-supervised Pre-training Models(GPT1_BERT).assets/image-20210315071610781.png" alt="image-20210315071610781" style="zoom: 67%;" />

사실 언어는 앞 뒤 문맥을 다 봐야하는 것인데, 기존의 Language Model은 왼쪽(전) 또는 오른쪽(후)만의 정보를 이용해왔다. 이런 맥락에서 등장한 것이 BERT의 pre-training 방식인 **`Masked Language Model(MLM)`**이다.

방식은 이렇다. 데이터가 주어지면, 이 문장 데이터 중 일정 비율(*k*%)을 `[MASK]` 토큰을 이용하여 Mask 처리하여, 나머지 단어들만을 가지고 이를 맞추는 연습을 한다. 이 때 **Masking 할 비율 `k`를 몇으로 할지는 사용자가 정하는 Hyperparameter**이다.

- Mask out *k*% of the input words, and then predict the masked words
- e.g. use *k* = 15%

- `k`가 너무 낮으면, 학습 시간 대비 훈련량이 적어 효율이 떨어지거나 학습 속도가 느려질 수 있다.
- `k`가 너무 높으면, 문맥을 제대로 파악할 수 없다.

문제는, pre-training 과정에서는 [MASK] 토큰이 어느 정도 비율로 있었는데, **실제 데이터에서는 [MASK] 토큰이 나올 일이 없다**는 것이다. 따라서 학습과정과 실제 데이터간의 괴리가 발생한다. 이런 차이점이 학습을 방해하거나 전이학습의 효과를 저해한다.



> **이를 해결하기 위한 방법으로** 마스킹할 *k*%의 단어들 전부가 아니라 어느 정도만 [MASK] 토큰으로 바꾸는 방식을 생각해볼 수 있다.

- 예를 들어, 전체 문장이 총 1000개의 단어이고 15%(150개) 마스킹 중 80%(120개)는 그대로 마스킹 토큰으로 두고, 10%(15개)는 random word로 변경한다.
  - 이는 마치 동사가 들어가야 할 자리에 이상한 명사가 들어가거나 하는 식의 문장이 될 것이다.
- 그러면 그 이상한 단어가 적용된 문장을 원래의 문장으로 복원할 수 있게 하는 방식으로, train 문제의 난이도를 더 올린다.

- 나머지 10%(15개)는 원래대로 둔다(모델은 이 단어들은 바뀔 필요가 없다고 예측해야 옳을 것이다.)



### Next Sentence Prediction

<img src="Lecture9_Self-supervised Pre-training Models(GPT1_BERT).assets/image-20210315071610781.png" alt="image-20210315071610781" style="zoom: 67%;" />



- **Next Sentence Prediction**은 **문장 level의 prediction** 기법이다.

- GPT-1에서 <Extract> token과 같은 Extract token(**정보 추출의 token**)으로  BERT에는 **<CLS> token**을 넣는다.
  - BERT에선 GPT와 달리 **문장의 가장 앞**에 넣어준다.
- 그리고 각 문장이 끝날 때 <SEP> 토큰을 넣어준다.
  - 그리고 이 sequence로 binary classification을 수행하여 문맥상 다음에 오는 문장이 맞는지 아닌지 학습한다.

- 문장 level에서는 **2개의 문장이 연속해서 나와도 어색하지 않은 문장**인지를 비교해줍니다.
  - 이는 **IsNext(연속 가능), NotNext(연속 불가능)**으로 비교판단합니다.

- 또한, BERT는 Sentence를 비교하면서 중간중간에 <MASK> token에 있는 값들을 예측을 수행하게 됩니다.



### BERT 요약

- BERT BASE : L=12, H=768, A=12    

- BERT LARGE : L=24, H=1024,  A=16

  - L : Self-attention블록 수
  - H : 각 Self-Attention에서 동일하게 유지되는 임베딩 벡터 차원수
  - A : num_heads

- WordPiece Embeddings (30,000 WordPiece)

  - BERT에서는 word단위의 Embedding vector를 사용하는 것이 아니라 word를 좀 더 잘게 쪼개서 subword 단위로 embedding 하는 방법을 사용했습니다. (WordPiece)

- Learned positional embedding

  - BERT에서는 Positional embedding 자체도 neural network 학습에 의해서 결정됩니다.

- [CLS] - Clasffication token

- Packed Sentence Embedding [SEP]

- Segment Embedding

  - 두 문장을 묶어 하나의 세그먼트로 만들 때, 몇번째 문장인지에 해당하는 세그먼트 임베딩을 추가하여 넣어줌으로써 문장을 구별할 수 있게 한다.

  - 예를 들어, 아래 그림처럼 <SEP> token을 통해 문장을 분리하는데 B 문장의 첫 단어 "he"를 구분해야 하지만, Positional Embedding은 단지 위치만 나타내는 벡터이기 때문에 A B를 구분할 수 없습니다.

  - 따라서 Segment Embedding을 추가함으로써 **A문장과 B문장을 나눠줍니다.**

    ![image-20210315072805164](Lecture9_Self-supervised Pre-training Models(GPT1_BERT).assets/image-20210315072805164.png)

- Pre-trained task

  - Masked LM
  - Next Sentence Prediction



### Fine-tuning Process

기존에 특정 Task를 처리하도록 pre-train된 모델을, 다른 task를 수행할 수 있도록 조정하여 주는 과정을 **`미세조정 과정(Fine-tuning Process)`**이라고 한다. 기존의 모델구조를 거의 바꾸지 않고 Output Layer만 바꾸면 되고 추가적인 아주 작은 조정만으로 다른 Task를 수행할 수 있기 때문에 엄청난 강점이 있다(성능 또한 굉장히 좋다)

![image-20210315072425465](Lecture9_Self-supervised Pre-training Models(GPT1_BERT).assets/image-20210315072425465.png)

​	(a) 문장 두개를 입력받아 분류(NSP, 논리적 모순 예측 등). [SEP]토큰이 추가된 걸 확인할 수 있다.

​	(b) 문장 하나를 입력받아 감정 등의 분류. 분류 문제의 경우 a,b 모두 [CLS]토큰 위치에서 예측값이 출력되는 것을 확인할 수 있다.

​	(c) Question Answering에 사용되는 경우

​	(d) 문장 구성성분의 품사를 예측하는 문제



### GPT-1 vs BERT

**`OpenAI GPT`**

- **Unidirectional -** 다음 단어를 예측하는것이 task이기 때문에, 다음 단어에 접근을 허용하면 안된다.
- BookCorpus 데이터로 학습(80억개 단어)
- 배치 사이즈 32,000개 단어
- 모든 fine-tuning experiments에 대하여 5e-5라는 동일한 학습률 적용

**`BERT`**

- **Bidirectional -** [MASK] 토큰으로 치환된 단어를 예측하기 위하여 앞뒤 문맥을 모두 사용한다.
- BookCorpus와 Wikipedia 데이터로 학습(250억개 단어)
- [SEP], [CLS], 세그먼트 임베딩
- 배치 사이즈 128,000개 단어
- task에 따라 각기 다른 학습률 적용

일반적으로 기존의 모델들에 비해 BERT가 성능이 전반적으로 좋았다(GLUE 자료 참조)

![img](https://media.vlpt.us/images/blush0722/post/73f3cc68-fe8b-4ff4-b08e-a89a2914a567/image.png)

> GPT는 주어진 sequence를 encoding 할 때 바로 다음 단어를 예측해야 하는 task를 진행하기 때문에 특정한 time step에서 그 다음에 나타나는 단어로의 접근을 허용하면 안 된다. 그래서 transformer의 decoder처럼 masked self attention을 사용한다.

> 
>
> 그러나 BERT의 경우 mask로 치환된 토큰들을 예측하기 때문에 mask 단어를 포함하여 전체 주어진 모든 단어들에 접근이 가능하다. 그래서 transformer의 encoder에서 사용되는 self attention module을 사용한다.





### 기계독해(MRC) 기반 질의응답(QA)

질의응답의 형태인데 질문만 주어지고 그 질문에 대한 답을 예측하는게 아니라 주어진 지문이 있을 떄 지문을 잘 이해하고 질문에서 필요로 하는 정보를 잘 추출할 수 있는 기계 독해에 기반한 질의응답이다.

![img](https://media.vlpt.us/images/blush0722/post/129a50f2-8d6e-489d-98cd-d88a9fc508fe/image.png)



### SQuAD

스탠포드에서 만든 질문과 답변에 대한 데이터 set으로 Stanford Question Answering Dataset의 약자이다. https://rajpurkar.github.io/SQuAD-explorer/ 이 링크를 통해 들어가면 SQuAD 2.0 버전과 1.1 버전에 대한 leaderboard 점수를 확인할 수 있다. (BERT라는 단어가 들어간 모델이 많이 보인다.)

#### SQuAD 1.1

지문에서 답을 찾을 수 있는 질문만 입력으로 주어진다.

![img](https://media.vlpt.us/images/blush0722/post/6226b1bf-9cde-4cb5-91a6-8ff78fe29022/image.png)

- main task를 수행하는 layer가 작동하는 방식
  먼저 질문과 지문을 <SEP> 토큰으로 이어주고 제일 앞에 <CLS> 토큰을 넣는다. 이 때 각 단어별 최종 encoding 벡터가 나오면 이를 공통된 output layer를 통해서 스칼라 값을 뽑도록 한다.

예를 들어 각각 encoding 벡터가 2차원으로 나오게 된 경우 output layer는 단순히 이 2차원 벡터를 단일한 차원의 스칼라 값으로 변경해주는 Fully Connected Layer가 된다. 이 때 Fully Connected Layer가 학습되는 파라미터가 된다. 각 스칼라 값을 얻은 후에는 softmax를 통과시켜 주고, answer가 시작하는 단어(위 문장에선 first)의 logic 값을 100%에 가까워 지도록 softmax loss를 통해 학습한다. 그리고 answer가 끝나는 단어를 예측하는 또 하나의 output layer를 통해 ground truth 단어를 예측하도록 학습한다.

![img](https://media.vlpt.us/images/blush0722/post/584666ad-912e-427c-be11-bd6a642c97fb/image.png)



#### SQuAD 2.0

지문에서 답을 찾을 수 없는 질문도 입력으로 주어진다.

- Use token 0 ([CLS]) to emit logit for “no answer”
- “No answer” directly competes with answer span
- Threshold is optimized on dev set

![img](https://media.vlpt.us/images/blush0722/post/7ded550d-1ab0-4b85-9afa-3b03a751d46d/image.png)

[CLS] 토큰을 binary classification을 통해 answer가 지문에 있는지 없는지를 판단하고, 없으면 'no answer'를 출력하고 있으면 SQuAD 1.1 처럼 답의 첫 문장과 끝 문장을 찾는다.



## On SWAG

- Run each Premise + Ending through BERT
- Produce logit for each pair on token 0 ([CLS])

**`SWAG`** 에서는 주어진 문장이 있을 때, 다음에 나타날 문장을 객관식으로 고르는 형태의 QA다. 이 경우에도 CLS 토큰을 사용한다.

- 4가지 선택지를 모두 하나씩 질문과 concat하여 벡터화하고, Fully Connected Layer를 통과시켜 예측 스칼라값으로 만든다.
- 4개의 스칼라값을 모두 softmax에 통과시켜, 정답에 해당하는 값의 확률을 높이도록 학습시킨다.

### Ablation Study

[알고리즘이나 모델의 feature를 제거하면서, 그 행위가 성능에 끼치는 영향을 평가하는 방식](https://fintecuriosity-11.tistory.com/73)을 Ablation Study라고 한다.

- **Big models help a lot**Layer를 점점 더 쌓고, 파라미터를 늘릴수록, 즉 Big model일수록 더 좋더라는 이야기.데이터셋이 3600개밖에 없더라도 파라미터를 110M→340M 개로 늘리니까 더 좋아졌다.GPU 리소스를 늘리면 늘릴수록 더 높아지더라. 점근선(asymptote)의 형태가 아니더라! 최대한 많이 리소스를 늘려라.

![img](https://media.vlpt.us/images/blush0722/post/fdde37f5-0745-4b9b-b05b-20f2053f0110/image.png)







































---

## Bert의 단점

BERT의 Masked Language Model의 단점은 무엇이 있을까요? 사람이 실제로 언어를 배우는 방식과의 차이를 생각해보며 떠올려봅시다



Byte2Piece를 했을 때,

장 점 이라고 나누었는데 '장'을 masking하면 굉장히 이상한 결과를 내게 된다.



pre-training 과정에서는 주어진 문장에서 평균적으로 15% 단어가 masking된 단어로 이루어져있는 문장에 익숙한 모델이 학습되는데, 이 모델을 main task에 수행할 때는 mask라는 토큰은 더이상 등장하지 않게 된다.
train에서 나오는 양상이나 패턴이 main task를 수행할 때 주어지는 입력에 대한 문장과는 다른 특성을 보여 학습을 방해하거나, 성능을 올리는데 방해요소가 될 수 있다.