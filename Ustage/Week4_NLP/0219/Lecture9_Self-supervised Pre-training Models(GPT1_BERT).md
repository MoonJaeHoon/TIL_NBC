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

![image-20210315055309507](Lecture9_Self-supervised%20Pre-training%20Models(GPT1_BERT).assets/img1.png)

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

![image-20210315064256429](Lecture9_Self-supervised%20Pre-training%20Models(GPT1_BERT).assets/img2.png)



# 2. BERT (Bidectional Trasnformer)

BERT는 현재 가장 많이 쓰이는 Pre-trained 모델이다. **<u>Masked</u>** **Language Modeling 방식으로 학습**되었으며, 그림상으로는 GPT와 아이디어 자체가 동일해보이지만 다르다.

GPT는 방향성이 없었다는 점을 기억해야 한다. 비슷한 구조 중에는 또한 양방향성 성질을 반영하여 LSTM을 기반으로 한 인코더 ELMO도 있었다.

하지만 현재 추세에서는 LSTM 인코더가 Transformer로 대체되면서 여러 강점을 가진 BERT가 가장 많이 쓰이고 있는 것이다.

![image-20210315064936702](Lecture9_Self-supervised%20Pre-training%20Models(GPT1_BERT).assets/img3.png)



## Masked Language Model (MLM)

<img src="Lecture9_Self-supervised%20Pre-training%20Models(GPT1_BERT).assets/img4.png" alt="image-20210315071610781" style="zoom: 67%;" />

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

<img src="Lecture9_Self-supervised%20Pre-training%20Models(GPT1_BERT).assets/img5.png" alt="image-20210315071610781" style="zoom: 67%;" />



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

    ![image-20210315072805164](Lecture9_Self-supervised%20Pre-training%20Models(GPT1_BERT).assets/img6.png)

- Pre-trained task

  - Masked LM
  - Next Sentence Prediction



### Fine-tuning Process

기존에 특정 Task를 처리하도록 pre-train된 모델을, 다른 task를 수행할 수 있도록 조정하여 주는 과정을 **`미세조정 과정(Fine-tuning Process)`**이라고 한다. 기존의 모델구조를 거의 바꾸지 않고 Output Layer만 바꾸면 되고 추가적인 아주 작은 조정만으로 다른 Task를 수행할 수 있기 때문에 엄청난 강점이 있다(성능 또한 굉장히 좋다)

<img src="Lecture9_Self-supervised%20Pre-training%20Models(GPT1_BERT).assets/img7.png" alt="image-20210315072425465" style="zoom: 200%;" />

​	(a) **Sentence Pair Classification Tasks :** 문장 두개를 입력받아 분류(논리적인 내포관계, 모순관계 등을 예측). [SEP]토큰이 추가되어 문장을 구분할 수 있게 해주고, [CLS] 토큰은 BERT 수행 후 Classification TASK를 수행할 때 사용되는 것을 확인할 수 있다.

<img src="Lecture9_Self-supervised%20Pre-training%20Models(GPT1_BERT).assets/img8.png" alt="image-20210315202610544" style="zoom:67%;" />

​	(b) **Single Senctence Classification Tasks** : 문장 하나를 입력받아 감정 등의 분류. 분류 문제의 경우 a,b 모두 [CLS]토큰 위치에서 예측값이 출력되는 것을 확인할 수 있다.

<img src="Lecture9_Self-supervised%20Pre-training%20Models(GPT1_BERT).assets/img9.png" alt="image-20210315202640407" style="zoom:67%;" />

​	(c) **Question Answering Tasks :** 주어진 질문에 답변을 하는 경우이다.

<img src="Lecture9_Self-supervised%20Pre-training%20Models(GPT1_BERT).assets/img10.png" alt="image-20210315202715169" style="zoom:67%;" />

​	(d) **Single Sentence Tagging Tasks :** 주어진 문장에서 문장 성분 혹은 품사를 예측하는 문제

<img src="Lecture9_Self-supervised%20Pre-training%20Models(GPT1_BERT).assets/img11.png" alt="image-20210315202725460" style="zoom:67%;" />



### GPT-1 vs BERT

**`OpenAI GPT`**

- **Unidirectional -** 다음 단어를 예측하는것이 task이기 때문에, 다음 단어(word)에 접근을 허용해선 안되는 구조이다.
- **Train Data size :** 80억개 단어(BookCorpus 데이터로 학습)
- **배치 사이즈 :** 32,000개 단어
- 모든 fine-tuning experiments에 대하여 `5e-5`라는 동일한 학습률 적용

**`BERT`**

- **Bidirectional -** [MASK] 토큰으로 치환된 단어를 예측하기 위하여 앞뒤 문맥을 모두 사용, 앞뒤 단어에 접근이 허용된다.
- **Train Data size :** 250억개 단어(BookCorpus와 Wikipedia 데이터로 학습)
- **배치 사이즈 :** 128,000개 단어
- [SEP], [CLS] Token을 사용
- 세그먼트 임베딩을 사용하여 여러 문장이 주어졌을 때 문장을 잘 구분할 수 있음.
- task에 따라 각기 다른 학습률 적용

일반적으로 기존의 모델들에 비해 BERT가 성능이 전반적으로 좋았다(GLUE Benchmark 참조)

![image-20210315200829619](Lecture9_Self-supervised%20Pre-training%20Models(GPT1_BERT).assets/img12.png)

> GPT는 주어진 sequence를 encoding 할 때 바로 다음 단어를 예측해야 하는 task를 진행하기 때문에 특정한 time step에서 그 다음에 나타나는 단어로의 접근을 허용하면 안 된다. 그래서 **transformer의 decoder처럼 masked self attention**을 사용한다.



> 그러나 BERT의 경우 mask로 치환된 토큰들을 예측하기 때문에 mask 단어를 포함하여 전체 주어진 모든 단어들에 접근이 가능하다. 그래서 **transformer의 encoder에서 사용되는 self attention module**을 사용한다.





### 기계독해(MRC) 기반 질의응답(QA)

질의응답의 형태인데 질문만 주어지고 그 질문에 대한 답을 예측하는게 아니라 주어진 지문이 있을 떄 지문을 잘 이해하고 질문에서 필요로 하는 정보를 잘 추출할 수 있는 기계 독해에 기반한 질의응답이다.

![image-20210315201432657](Lecture9_Self-supervised%20Pre-training%20Models(GPT1_BERT).assets/img13.png)



### SQuAD

스탠포드에서 만든 질문과 답변에 대한 데이터 set으로 Stanford Question Answering Dataset의 약자이다. https://rajpurkar.github.io/SQuAD-explorer/ 이 링크를 통해 들어가면 SQuAD 2.0 버전과 1.1 버전에 대한 leaderboard 점수를 확인할 수 있다. (BERT라는 단어가 들어간 모델이 많이 보인다.)

#### SQuAD 1.1

지문에서 답을 찾을 수 있는 질문만 입력으로 주어진다.

![image-20210315205648441](Lecture9_Self-supervised%20Pre-training%20Models(GPT1_BERT).assets/img14.png)



> main task를 수행하는 layer에서는 어떻게 작동할까?



1. 먼저 질문과 지문을 <SEP> 토큰을 통해 하나의 seq로 concat한 다음, 제일 앞에 <CLS> 토큰을 넣습니다.

2. 그 후 인코딩을 진행하여  각 word별로 최종 encoding 벡터가 Output으로 나왔을 때, 이를 공통된 output layer를 통해서 **scalar 값**을 뽑도록 합니다. 
   - 즉, encoding vector가 나온 후, 이 vector들을 **scalar값**으로 변경해준다는 것입니다.
3. scalar값을 각 word별로 얻은 후에는 여러 word들 중에 답에 해당하는 문구가 어느 단어에서 시작하는지 **fully connected layer** <u>***FC1***</u>를 통해 먼저 예측해줍니다.
   - 예를 들어서, 만약 124개의 단어가 존재하면 (이들을 각각 워드벡터로 인코딩하고 scalar로 변환하면) 124개의 scalar값의 vector가 도출되며
   - first라는 단어가 정답의 시작에 해당하는 단어이기 때문에 단어별 scalar들 중 "first" 단어에서 softmax의 높은 확률을 가질 수 있도록 **loss**를 통해 학습합니다.
4. 이후에는 Answering 단어가 끝나는 시점도 예측해주어야 하는데 이 word에 대한 또 다른 **fully connected layer** ***<u>FC2</u>***를 만들고 Softmax를 통과해서 높은 확률의 end 위치를 설정해줍니다.
   - 위 그림에서는 `shock`이 end위치에 해당하며 역시 이것도 Loss를 통해 학습되게 된다.

![image-20210315210723588](Lecture9_Self-supervised%20Pre-training%20Models(GPT1_BERT).assets/img15.png)



#### SQuAD 2.0

SQuaAD 2.0에는, SQuaAD 1.1에다가 `지문에서 답을 찾을 수 없는 질문 Dataset`도 입력으로서 추가되어있습니다.

이런 경우에는 BERT의 Fine-Tuning에서 답이 있는지 없는지를 예측하는 Task가 추가되어야 합니다.

만약 정답이 존재한다면 다음 과정을 수행하고(SQuAD 1.1처럼 진행한다), 그렇지 않으면 종료를 합니다.

- 질문과 문단을 종합적으로 보고 판단하기 때문에 CLS 토큰을 활용합니다.
- 이것 역시 질문과 문단을 concat해서 BERT로 인코딩하여 CLS 토큰을 얻는 것입니다.
- <CLS> 토큰을 이진분류하는 OutputLayer에 통과시켜 "answer"(정답이 존재), "no answer"(정답이 없음)을 구분합니다. (크로스 엔트로피로 학습합니다.)
- 정답이 존재하지 않는다면, "no answer"라면 종료한다.

<img src="Lecture9_Self-supervised%20Pre-training%20Models(GPT1_BERT).assets/img16.png" alt="image-20210315211428484" style="zoom: 80%;" />

- 그리고 만약 "answer"(정답이 존재)라면 SQuAD 1.1을 수행한 방식대로 하면 됩니다. (정답의 첫 단어와 끝 단어를 찾는 과정을 수행)



## On SWAG

**On-SWAG**는 주어진 문장**이 있을 때, **다음에 나타날 법한 문장을 객관식으로 고르는 task 형태의 QA입니다.

- **주어진 문장**과 **(i)**를 <SEP>토큰 이용 concat 해서 BERT 통해 encoding 하고, 나오는 <CLS> token을 사용해서 encoding vector를 fully connected layer를 통해 **scalar값**을 추출합니다.
  - 사실 <CLS> 토큰은 pre-trained task 중 Next Sentence Prediction을 수행할 때 입력 seq의 context를 담은 tokne으로서 활용됩니다.
  - 이렇게 seq의 정보를 담고 있기 때문에 OutputLayer의 입력으로 사용되는 것이죠.
- 문장들 (ii), (iii), (iv)도 각각 질문과 concat해서 벡터로 만든 뒤 역시 FC layer를 거쳐 scalar값을 구해냅니다.
  - 이것 역시 Sentence Pair인 경우에 해당하므로, <CLS> 토큰의 output hidden state를 Output Layer의 입력으로서 사용하는 과정을 수행합니다.
- 이렇게 나온 4개의 scala value를 모두 softmax에 통과시켜서 정답에 해당하는 부분의 확률이 높도록 학습시킵니다.
  - 각 pair 별로 각 <CLS> 토큰에 logit 값을 생성해주는 것입니다.

![image-20210315213738884](Lecture9_Self-supervised%20Pre-training%20Models(GPT1_BERT).assets/img17.png)

## Ablation Study

알고리즘이나 모델의 feature를 제거하면서, 그 행위가 성능에 끼치는 영향을 평가하는 방식을 Ablation Study라고 한다.

- Big Model 일수록 (Layer를 더 많이 쌓고 파라미터를 더 많이 늘릴수록) 무조건적으로 더 좋다는 결과를 보여주고 있다.
  - 데이터셋이 3600개밖에 없는데도 parameter 수를 110M에서 340M으로 끌어올렸을 때가 더 좋아졌다.
  - 아래 그림처럼 증가하는 경향이 점근선도 없는 형태더라.
- 그렇기 때문에 GPU 자원을 최대로 이용할 수 있을 때까지 model 크기를 늘릴 수 있다면 늘리는 것이 좋다고 전망된다.

![image-20210315221050893](Lecture9_Self-supervised%20Pre-training%20Models(GPT1_BERT).assets/img18.png)











---

## +추가정리(Bert의 단점)

BERT의 Masked Language Model의 단점은 무엇이 있을까요? 사람이 실제로 언어를 배우는 방식과의 차이를 생각해보며 떠올려봅시다



Byte2Piece를 사용했을 때,

장 점 이라고 나누었는데 '장'을 masking하면 굉장히 이상한 결과를 내게 된다.



pre-training 과정에서는 주어진 문장에서 평균적으로 15% 단어가 masking된 단어로 이루어져있는 문장에 익숙한 모델이 학습되는데, 이 모델을 main task에 수행할 때는 mask라는 토큰은 더이상 등장하지 않게 된다.
train에서 나오는 양상이나 패턴이 main task를 수행할 때 주어지는 입력에 대한 문장과는 다른 특성을 보여 학습을 방해하거나, 성능을 올리는데 방해요소가 될 수 있다.