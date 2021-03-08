## 1. ALBERT에서 shared에 대해 질문이 있습니다.

팀원들 끼리 shared 되는 weight에 대해 의견이 다양하여 질문을 드립니다.

1. 동일한 layer를 쌓아(예를들어 동일한 encoder layer를 6번 통과 시킨다) layer간의 weight를 share한다. 

2. 한 layer 내에서 weight를 공유한다. 

1번 방법의 경우 이 부분이 참이 되려면 ALBERT의 weight 표에서 모든 가중치를 공유했을 때 3배가 아니라 그 이상이 줄어야 할 것으로 예상됩니다. 그래서 2번 방법이 유력하다고 생각이 되는데 어떤 방식으로 공유가 되는지 수식이나 참고자료를 알 수 있을까요?

 

> 1번이 맞습니다. 질문해주신 ALBERT의 weight 표는 혹시 어디서 나온 부분인지 여쭤봐도 될까요? 제가 확인해본 바론 ALBERT 논문의 Table 1 기준으로 1/10 미만으로 줄어든 것으로 확인됩니다. 만약 Table 3을 말씀하신거라면 여긴 embedding size를 base 버전 ALBERT보다 더 크게 잡았기 때문에 그렇게 보이는 것이라고 생각합니다.





## 2. GPT-2 scaled factor

GPT-2에서 GPT-1과 다른 부분을 설명하실 때,

output 값에 가까운 layer들을 initalize할 때, 점점 작은 값으로 initialize한다고 하셨습니다.

그 이유는 위의 레이어의 역할이 줄어들도록 하기위해서라고 설명을 해주셨는데요,

<img src="(10강) Advanced Self-supervised Pre-training Models.assets/_2021-02-22__12.04.06.png" alt="img" style="zoom: 50%;" />



왜 이러한 scaled intialize를 하는지 좀더 자세히 알고 싶습니다!



>GPT-2 논문에는 scale했다고만 나와있고 별도로 왜 이런 접근을 했는지 명시되어 있진 않지만, 원리는 어렵지 않게 유추해볼 수 있습니다. GPT-2는 보시면 아시겠지만 굉장히 layer를 깊게 쌓은 큰 모델입니다. 딥 러닝 모델 내의 모든 계산은 그 기반은 전부 matrix multiplication으로 이루어져 있고 이에 따라 곱셈과 덧셈이 몇 백 차원에 걸쳐 계속 반복적으로 이루어지도록 되어 있습니다. 이 때문에 계속해서 각 element들의 값의 분산 값이 점점 달라지게 되고 이 때문에 값이 지나치게 explode 해버리거나 vanishing해버릴 수 있는 문제가 있습니다. 즉, 아래에서 위로 갈 수록 각 layer의 연산을 거치면서 이러한 현상이 계속 누적이 되죠. 그래서 실제로 위로 갈 수록 이러한 weight들의 영향력을 줄여 값을 어느정도 안정화해주기 위함이라고 생각합니다.
>
> 
>
>Initialization에 따른 exploding/vanishing에 대한 예시는 아래 링크에 잘 설명되어 있습니다.
>
>https://towardsdatascience.com/weight-initialization-in-neural-networks-a-journey-from-the-basics-to-kaiming-954fb9b47c79





## 3. GPT3의 one shot, few shot

GPT3의 one shot, few shot에 대해 궁금한 점이 있습니다.

교수님께서 GPT3는 zero shot, one shot, few shot으로 원하는 테스크를 수행할 수 있다고 하시면서,

한개 또는 적은 수의 task에 적합한 dataset을 넣어주면, 더 좋은 결과를 얻을 수 있다고 하셨습니다.



그러면 제공한 적은수의 dataset은 어떻게 활용이 되어서 더 좋은 성능을 보일 수 있는 것인가요? 적은 수의 dataset만으로 학습을 하고, back prop이 되면서 원하는 task에 적절한 weight를 갖게 되어 더 좋은 성능을 보이는 것인가요?

( 제 생각에는 너무 적은 dataset은 오히려 큰 noise로 작용하지 않을까하는 의문이 들었습니다.)



> **송재우 조교님**
>
> 제가 GPT-3에 대해서 자세히 읽어본 것은 아니지만 제가 알고 있는 관점에서 말씀을 드리겠습니다.
>
>  
>
> BERT 논문의 5.2 section에서 BERT의 연구진들은 모델이 매우 크고 충분히 pre-train된다면 설사 fine-tuning data가 매우 적다고 해도 적은 수의 random initialized parameter만으로도 충분히 좋은 성능을 낼 수 있다고 주장합니다. 즉, 말씀하신대로 적은 데이터는 분명 noise처럼 작용할 수 있지만 BERT나 GPT와 같이 충분히 큰 모델이 충분히 많은 데이터로 미리 language에 대한 지식을 충분히 학습해두었다면 이러한 적은 수의 데이터가 모델에 가하는 noise의 효과보다 모델의 좋은 representation이 소량의 fine-tuning data의 정보와 맞물려 낼 수 긍정적인 효과가 더 크다는 의도로 보입니다.(어디까지나 저자들은 hypothesis라고 하지만요) 그래서 해석해보면 BERT가 워낙 크고 parameter가 많아서 소량의 데이터가 주는 noise는 큰 문제가 되지 않을 정도로 model이 robust하다는 뉘앙스로 보입니다.
>
>  
>
> GPT-3의 zero, one, few-shot learning도 같은 맥락에서 이해해볼 수 있다고 생각합니다. GPT-3는 BERT보다 훠얼씬 큰 모델이며 훠얼씬 많은 데이터로 학습이 되었기 때문에 그 자체로도 이미 LM 성능은 충분히 좋다고 볼 수 있는데요, 여기서 원하는 task에 대한 아주 극소량의 데이터만 넣어줘도 이미 모델 자체로 가지고 있는 충분한 언어 이해력과 표현력으로 이러한 noise를 극복하고 시너지를 낼 수 있다는 뜻으로 보입니다.



> **김태희 조교님**
>
> GPT-3는 처음 pre-train 이후에 parameter를 업데이트하는 과정은 없습니다.
>
> prompt라고 하는 앞으로 들어올 데이터셋을 어떤식으로 풀어야하는 지에 대한 guide text를 one-shot, few-shot으로 입력 문장의 앞 부분에 넣어주기만 해도 zero-shot에 비해서 엄청나게 성능이 많이 오르고 이 부분에 대해 많은 연구자들이 놀랐다고 생각합니다 ㅎㅎ 기존의 one-shot, few-shot과 조금 내용이 달라서 내용이 헷갈리실 수 있을 것 같습니다
>
> 이런 식으로 모델이 활용되는 경우가 없었어서 나름 고민을 해본 적이 있었는데요. 재우 조교님과 유사하지만 결론은 gpt-3가 아주 많은 양의 text를 학습하면서 prompt로 제공된 text의 상징적/사전적 의미를 아주 잘 파악할 수 있는 모델이 되었기 때문이라고 생각합니다.
> 우리가 입력한 prompt의 형태는 이미 사전 학습된 text에서 무수히 많은 유사한 pattern일 것이고 이런 식으로 text를 입력해주는 것만으로 어느정도 성능이 오르는게 아닐까 생각합니다. translation을 하는 것이든 qa를 하는 것이든 task description과 "=>" 을 포함한 몇개의 example을 통해 어떤 text를 generation해야할 지 추론할 수 있는 것이고 기존의 모델들은 parameter를 업데이트해야하고(fine-tuning) 훨씬 더 많은 example이 필요했다는 것이 차이점일 것 같습니다
>
> (논문 참고: https://arxiv.org/pdf/2005.14165.pdf) 
>
> 
>
> <img src="(10강) Advanced Self-supervised Pre-training Models.assets/mceclip0.png" alt="img"  />





## 4. batch size 관련 질문드립니다.

GPT-2 에비해 GPT-3의 배치사이즈가 늘어났고 모델 성능이 더 좋아졌는데, 이 전 DL Basic 강의에서 batch size 가 커지면 sharp minimum 를 형성하여 generalization performance 가 안 좋아진다는 설명을 들었었는데, 혼동이 되어 질문드립니다.



일반적으로 리소스가 허용이 되는한 배치사이즈를 늘려서 학습하는게 좋은건가요? 그렇다면 generalization에는 영향을 안끼치는건가요?



> **David 조교님**
>
> TL;DR : Batch Size가 커져서 성능이 좋아진 것이 아닌, Larger Model이 되어 성능이 좋아진 것입니다. (1.5B -> 175B) 그리고, 더 복잡한 task [larger model, difficult data]을 "효율적"으로 학습하기 위해서 더 큰 Batch Size (gradient noise scale 계산에 따르면 - https://arxiv.org/pdf/1812.06162.pdf) 가 필요합니다.
>
>  
>
> Batch size를 키우면, 학습 속도를 키울 수 있다는 장점이 있습니다... 말씀하신 generalization tradeoff도 있지만요.
>
> GPT 같이 큰 모델을 학습시킬 때는 Computation Cost (time-computation tradeoff) 도 고려가 되어야 합니다. ( GPT 2 학습에 5,100만원이 든다! - deview 2019 큰 언어 모델 가동기 https://deview.kr/2019/schedule/291 ) 그래서, [ An Empirical Model of Large-Batch Training https://arxiv.org/pdf/1812.06162.pdf ] OpenAI에서 해당 실험을 진행하여 어느정도 크기의 배치 사이즈가 효율적인지 계산하였습니다. 해당 논문에 따라 GPT3의 배치 사이즈를 증가시켰습니다.



> **캠퍼분**
>
> 안녕하세요. 저희 조에서 관련한 내용으로 얘기한 적이 있었고, 그 때 좋은 글을 하나 찾았어서 링크 올려드립니다! 조교님께서 답변해주시기 이전에 먼저 참고해보시면 좋을 것 같아요~ 말씀하신 generalization performance 관련한 내용 비롯하여 배치사이즈를 키우는 이유에 대하여 결론부에 나와있습니다.
>
> https://hongdoki.github.io/2017/10/07/optimization-difficulty-and-generlization-performance-as-batch-size-increases.html

