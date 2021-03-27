## 1️⃣ **NLP 종류**

 

NLP는 **Natural language processing**의 약자로 사람들이 일상적으로 쓰는 언어를 뜻합니다.

NLP는 컴퓨터가 주어진 단어나 문장, 긴 문단을 이해하는 **Nautral Language Understanding (NLU)**와 자연어를 생성하는 **Nautral Language Generating (NLG)**로 구성됩니다.

 

 

**1) Natural language processing**

 

- Low-level parsing

  - **Tokenization : 토큰화 ( token이라 불리는 단위로 나누는 작업을 뜻함)**

    token의 개념은 의미, 단어 등 여러 가지로 사용될 수 있습니다.

    

  - **Stemming : 다양한 의미변화를 없애고 그 의미만을 보존하는 단어의 어근을 추출하는 것**

    언어는 어미의 변화가 변화무쌍합니다. 예를 들어,
    하늘이 맑다
    하늘이 맑은데
    하늘이 맑지만
    하늘이 맑고
    다양하게 변할 수 있습니다.
    이러한 수많은 어미의 변화 속에서도 그 단어들이 모두 같은 변화를 나타낸다는 것을 컴퓨터가 이해할 수 있어야 합니다.

    

- Word and pharse level

  

  - **NER :** **단일 단어 혹은 여러 단어로 이루어진 고유명사를 인식하는 task

    **

  - **POS :** **Word들이 문장 내에서의 품사(부사, 형용사, 명사)나 성분이 무엇인지 알아내는 task**

  - **Noun-phrase chunking

    **

  - **Dependency parsing
    **

  - **Coreference resolution

    **

- Sentence level

  - **Sentimentation analysis : 가진 문장이 긍정 또는 부정인지 예측하는 task

    **ex) I like movie - 긍정, I hate movie - 부정, This movie is not bad - 부정 아님

    

  - **Machine translation : 번역**

    어순을 고려하는 것이 중요

    

- Multi-sentence and paragraph level

  - **Entailment prediction : 두 문장 간의 논리적인 내포 혹은 모순 관계를 예측

    **"어제 John이 결혼을 했다" , "어제 최소한 한 명은 결혼을 했다"
    두 문장 간에는 첫 문장이 참인 경우에 두 번째 문장은 자동으로 참입니다.

    하지만 "어제 한 명도 결혼하지 않았다"는
    "어제 John이 결혼을 했다"와 양립할 수 없는 논리적으로 모순관계를 가지게 됩니다.

    

  - **Question answering

    **ex) Google에서 예전에 문장이 들어간 정보만 찾아주었다면, 질문의 의도를 정확히 파악하고 검색 결과를 정확히 나타내 주는 것

    

  - **Dialog systems

    **ex) 챗봇과 같이 대화를 수행

  - **Summarization**

    ex) 주어진 문장 (뉴스 등)을 한 줄 형태로 요약

 

 

**2) Text Mining**

 

-  **빅데이터 분석과 연관

  **텍스트 및 문서 데이터에서 유용한 정보를 추출합니다.
  **
  **

- **Document clustering (Topic modeling, 문서 군집화)**

  다른 의미이지만 비슷한 의미를 가지는 keyword를 그룹핑해서 분석해야 되는데 이 것을 자동화해서 사용하는 기법

  

- **Social science와 연관

  **SNS, Media data 등에서 사람들의 성향을 파악할 수 있습니다.

 

 

**3) Information retrieval (정보 검색)**

 

- **구글이나 네이버에서 사용되는 검색 기술을 주로 연구하는 분야

  **

- **추천 시스템으로 진화**

 

 

 

## 2️⃣ **Bag-of-Words**

 

### **1) Bag-of-Words**

 

Bag-of-Words는 Text mining 분야에서 딥러닝이 적용되기 이전에 많이 활용되던 **단어 및 문서를 어떤 숫자 형태로 나타내는 가장 간단한 기법**입니다.

 

- **STEP 1 :** **사전에 unique word를 등록**



![img](https://blog.kakaocdn.net/dn/ufM4s/btqXu3Z3IdX/bWoJLxrsikvGg0ZDZkyRUK/img.png)



 

- **STEP 2 : One-hot vector로 encoding
  **



![img](https://blog.kakaocdn.net/dn/bkhvdN/btqXrDgjgrM/vMKpkv1CKfJTVtYSOUqqRk/img.png)



 

- **STEP 3 : Bag-of-Words vector (\**T\**he sum of one-hot vectors)로 나타냄**



![img](https://blog.kakaocdn.net/dn/px0Zv/btqXgpwqUM7/McIi0j17VfptGxFhKk4nyk/img.png)



 

One-hot vectors의 합을 Bag-of-words vector라고 부르는 이유는 vocabulary 상에 존재하는 각 word별로 bag를 준비하고 특정 문장에서 나타난 word들에 순차적으로 bag에 넣어준 후, 최종적으로 각 차원에 해당하는 word들의 수를 세서 최종 vector로 나타내기 때문입니다.

 

 

### **2) NaiveBayes Classifier**

 

이런 Bag-of-Words vector를 정해진 category 또는 Class로 분류하는 작업을 NaiveBayes Classifier라고 합니다.

 

NaiveBayes Classifier를 구하는 식은 아래와 같습니다.

 

- **cc :** class 개수
- **dd :** document
- **P(c|d)P(c|d) :** document가 주어졌을 때, 해당 class cc가 속해있을 확률
- **P(d)P(d) :** 문서 dd가 뽑힐 확률

 

P(c|d)P(c|d)가 가장 높은 값을 가졌을 때 그때의 cc를 택하는 방법입니다. Bayes Rule을 사용했습니다.

 



![img](https://blog.kakaocdn.net/dn/cAgQKU/btqXrCV1qXy/KZnJHn84EaW5mCeOl1kknK/img.png)https://www.edwith.org/bcaitech1



 

P(d)P(d)는 특정 확률분포를 나타내며, 상수로 나타낼 수 있기 때문에 무시할 수 있습니다.

 

P(d|c)P(c)P(d|c)P(c)를 아래 식과 같이 표현할 수 있습니다.

이때, P(d|c)P(d|c)를 단어들의 Sequence P(w1,w2,....,wn|c)P(w1,w2,....,wn|c)로 표현할 수 있으며, 

**cc가 고정되어 있고 w1,w2,....,wnw1,w2,....,wn이 모두 독립이라고 가정했을 때, 곱의 형태로 표현**할 수 있습니다.

 



![img](https://blog.kakaocdn.net/dn/bpIP5Z/btqXrDUWEQw/okiwZDYF4vuWJBKssGTkW1/img.png)



 

예를 들어, 아래와 같은 document와 class가 존재할 때, "Classification task uses transformer"라는 문장의 class cc를 구하는 문제입니다.

 



![img](https://blog.kakaocdn.net/dn/bW6o69/btqXj8urTEJ/cSo6geCKIx1a7qg89CkmO0/img.png)https://www.edwith.org/bcaitech1



 

Class의 확률분포 P(ccv),P(cNLP)P(ccv),P(cNLP)는 다음과 같습니다.

(각각의 class는 2개, 2개로 등장을 할 수 있기 때문에 1/2, 1/2)

 



![img](https://blog.kakaocdn.net/dn/bElnFq/btqXb1bAVDu/fKLTFEAYXKtuJ7ZmvAvCs1/img.png)



이제 각각의 Class에 대해 Word의 확률분포를 구합니다.

CV Class에서는 task는 한 번만 등장했기 때문에 1/14로 추정할 수 있습니다. (14는 CV Class의 총 word 수)

하지만, NLP Class에서는 task가 두 번 등장했기 때문에 2/10로 추정할 수 있습니다. (10은 NLP Class의 총 word 수)

나머지 확률도 구하면 아래 표와 같습니다.

 



![img](https://blog.kakaocdn.net/dn/IL6MZ/btqXu4YYWut/dW7guIMRfLk8TyQUs4mk7K/img.png)https://www.edwith.org/bcaitech1



 

각 word가 독립적이라고 가정했을 때, document d5d5 " “Classification task uses transformer"는 어떤 Class에 속하는지 추론할 수 있습니다.

 



![img](https://blog.kakaocdn.net/dn/1Z9wJ/btqXrDm34h1/UIvUoB8Qkq8rsWu8Jqc9j0/img.png)https://www.edwith.org/bcaitech1



 

 

 

## 3️⃣ **Word Embedding**

 

Word Embedding은 **각 단어들을 vector로 변환해주는 기법**입니다.

Word Embedding에서 중요한 점은 **비슷한 의미를 가지는 word가 좌표상에서 비슷한 위치의 vector로 mapping** 되도록 표현해야 합니다.

ex) cat, kitty는 short distance / cat, hamburger는 far distance

 

### **1) Word2Vec**

 



![img](https://blog.kakaocdn.net/dn/xT1HM/btqXb1bA6IK/5N6aXswsN8ZSE96mnZtnF1/img.png)Distributed Representations of Words and Phrases and their Compositionality, NeurIPS’13



 

Word2Vec은 word embedding 방법 중 가장 대표적인 방법입니다.

Word2Vec은 비슷한 문맥의 단어들은 비슷한 의미를 가진다는 가정을 가집니다.

 

예를 들어, 아래 두 개의 문장이 있습니다.

> The cat purrs.

> The cat hunts mice.

cat은 "The", "purrs" / "This", "hunts", "mice"와 높은 관련성을 가진다는 것을 알 수 있습니다.

따라서 cat 주변에 나타나는 단어들의 확률분포를 예측할 수 있습니다.

 



![img](https://blog.kakaocdn.net/dn/7Mjda/btqXobYGOBE/20Jm1dWyTtLrELGBUhxIN0/img.png)Distributed Representations of Words and Phrases and their Compositionality, NeurIPS’13



 

 

Word2Vec 알고리즘은 다음과 같은 순서로 동작합니다.

 

- **STEP 1 : Sentence tokenization 수행

  **ex) Sentence : "I study math."

  

- **STEP 2 : Unique 한 단어들만 모아서 vocabulary 생성

  **ex) Vocabulary : {"I", "study", "math"}

  

- **STEP 3 : One-hot vector로 변경

  **ex) "study" [0,1,0]

  

- **STEP 4 : Sliding window를 사용해서 한 word를 중심으로 앞 뒤로 나타난 각각의 word와 입출력 단어 쌍을 구성

  **ex) I => (I, study)**
  **study => (study, I), (study, math)
  math => (math, study)

  

- **STEP 5 : 입출력 단어 쌍들에 대해 예측 task를 수행하는 neural network 생성

  **입력은 one-hot vector로 들어가게 됩니다.
  hidden layer의 개수는 hyper parameter(사용자가 정함)
  만약, Input vector가 [0,1,0]이면 => hidden layer는 2개 => ouptut layer는 3개 형태로 나올 수 있습니다.

 



![img](https://blog.kakaocdn.net/dn/by1CWL/btqXj8Bc8jH/J0o2dpEIF7aCwkyPgzZ9z0/img.png)Distributed Representations of Words and Phrases and their Compositionality, NeurIPS’13



 

 

Word2Vec은 유사한 word끼리 유사한 방향을 보입니다.

즉, vector의 방향은 비슷한 관계성을 보인다는 것을 알 수 있습니다.

 

- Man -> Woman
- Uncle -> Aunt
- King -> Queen

위의 vector들 모두 다 남성과 여성의 관계를 나타낼 때 같은 방향을 보이는 것을 알 수 있습니다.

 



![img](https://blog.kakaocdn.net/dn/XzNnE/btqXj8BddP2/RmiZoosGfaSMbDqL0RHHd0/img.png)



 

### **2) Glove**

 

Glove의 특징은 입출력 단어 쌍들에 대해서 **학습 데이터의 두 단어가 한 윈도우 내에서 총 몇 번 동시에 등장했는지를 사전에 미리 계산**합니다. 이를 통해, 동일한 단어 쌍에 대한 훈련을 반복적으로 하는 것을 피합니다.

 

따라서 Glove는 새로운 Loss function을 사용합니다.

입력 word의 embedding된 **vector uiui**와 출력 word의 embedding **vector vjvj**의 **내적** 값과,

**한 윈도우 내에서 동시에 몇 번 나타났는가**를 나타내는 **PijPij**에 loglog값을 취해서,

**uivjuivj와 logPijlogPij의 값이 최대한 가까워질 수 있도록 loss function 사용**합니다.

 



![img](https://blog.kakaocdn.net/dn/UfWUB/btqXj8usCcl/7MdWsYN1meykLK6Qqkpuf0/img.png)GloVe: Global Vectors for Word Representation, EMNLP’14



 

Glove 모델을 통해 vector들의 방향을 따져봤을 때 **비슷한 관계를 가지는 단어들은 비슷한 모양을 가진다는 것**을 알 수 있습니다.



![img](https://blog.kakaocdn.net/dn/clhIEs/btqXj8gQCCW/6Eur8o9iEl7NR63kXJrzI0/img.png)