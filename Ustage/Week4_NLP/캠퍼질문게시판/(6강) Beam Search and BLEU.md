## 1.Beam search 관련해서 질문드립니다.

[![img](https://cphinf.pstatic.net/mooc/20210217_240/16135527239919FPSn_PNG/day18_img.png)](https://cphinf.pstatic.net/mooc/20210217_240/16135527239919FPSn_PNG/day18_img.png)

Beam search와 관련해서 궁금한 점이 생겨 질문드립니다. Beam search를 진행할 때 스코어가 높은 k개의 경우만을 유지하면서 다음 단어를 찾는다고 배웠습니다. 이 때 루트마다 완성된 문장의 길이가 서로 다를 수도 있기 때문에 먼저 완성된 문장은 보관하고, 아직 완성되지 않은 문장에 대해 진행을 이어나간다고 들었습니다.

여기서 완성된 문장을 보관한 후, 남은 루트를 진행할 때 Beam size인 k를 유지하면서 진행을 하나요?

만일 위의 그림에서 I-was-hit 문장이 높은 스코어를 기록해서 보관이 되었다고 가정한다면, 이후 남은 루트 중 스코어가 가장 높은 he-struck-me 루트가 진행이 될 거라고 생각합니다. 이때 Beam size인 2를 그대로 유지하며 with 루트와 on 루트를 진행하는지, 아니면 이미 한 개의 문장을 보관했기 때문에 가장 스코어가 높은 with 루트만 진행하는지 궁금합니다.



> **김태희 조교님**
>
> 완성된 문장에 대한 결과값은 저장하고 남은 루트를 진행할 때 beam size를 계속 유지하면서 진행하는게 맞을 것 같습니다. 완성된 문장은 저장이 되면 마지막에 결과값 비교만 하면 되기 때문에 다른 문장들이 decoding되는 과정에서는 cost가 추가로 발생하지는 않습니다. (메모리는 잡아먹지만요) beam search의 목적상 beam size인 k를 유지하며 후보 문장을 탐색하는 것 같습니다.



> **David 조교님**
>
> 남은 루트 진행 시 여러 탐색 전략이 있는데, Node를 끝까지 진행해도 되고, 스코어 합산 값이 특정 이하 (상대적/절대적) 면 계속 진행하는 방식이 있다고 하네요. 아래 논문에서 pruning (어디에서 끊는지 결정) 방법에 따른 성능 비교 실험을 확인해보셔도 좋을 것 같습니다.
>
> ref:  https://arxiv.org/pdf/1702.01806.pdf ( Beam Search Strategies for Neural Machine Translation )



## 2. BLEU Score 관련 질문이 있습니다.

BLEU Score를 계산할 때, 1gram 부터 4gram까지 계산하여 곱한 후 4중근을 취하는 형식은, 1gram ~ 4gram 중 하나라도 0이 발생하는 경우 BLEU score = 0 이 되는 문제가 생기는데요.

만약 길이가 5~6 단어 정도로 짧은 문장에서 4-gram 이 하나라도 매치되지 않는 경우를 생각해보면 BLEU score가 0이 나오는데, 이는 괜찮은 현상인가요? 혹은 길이가 짧은 문장에서는 BLEU score를 계산할 때 2gram이나 3gram까지만 고려할 수도 있나요? 

예를 들어,

Reference : "I love you very very much."

Prediction : "I love you so much."

1gram : 4/5

2gram : 2/4

3gram : 1/3

4gram : 0/2

BLEU score = 0

이 되는 문제가 발생하지만, 실제 prediction과 reference는 의미상 거의 차이가 없다고 생각합니다.



> 네 관련해서 여러가지 normalization/smoothing 방법이 있고 몇가지 방법이 nltk에 구현되어 있습니다. 
>
> (참고: https://www.nltk.org/_modules/nltk/translate/bleu_score.html) 
>
> 예를 들면, n-gram 결과가 0인 경우 아주 작은 값을 갖게 하거나, 분자와 분모에 무조건 1을 더해줄 수 있습니다. 관련 technique들이 소개된 논문을 첨부해 드립니다. 
>
> (A systematic Comparisons of Smoothing Techniques for Sentence-level BLEU: https://www.aclweb.org/anthology/W14-3346.pdf)
>
> 
>
> +참고로 실제로 nltk의 bleu_score를 사용하지는 않고 최근에는 sacrebleu라는 metric을 사용하는 추세입니다
>
> (sacrebleu paper: https://www.aclweb.org/anthology/W18-6319.pdf).

