## 1. RNN cell state 업데이트 관련해서 질문드립니다.

"한 번의 선형 변환 만으론 Ct-1에 더해줄 정보를 만들기 어려운 경우 →

더해주고자 하는 값보다 좀 더 큰 값으로 구성된 정보를 형성 = C {tilde} →

그 값에서 특정 비율만큼의 정보를 덜어낸 후 Ct-1에 더해줌"

라는 식으로 설명해주셨습니다. 선형변환한 값에 tanh함수를 씌어주고 input gate를 곱한 값이 왜 더 큰 값으로 구성되었다는건지..(input gate 값은 0~1 사이의 값인 것 같은데) 혹시 관련된 내용이 첨부된 논문이나 수학적 증명이된 블로그 같은게 있는지 알 수 있을까요?



> 슬라이드 하단에 첨부된 링크를 참고하시면 될 것 같습니다. 그리고 해당 부분은 LSTM 원문에 기술되어 있으니 해당 내용을 참고하시면 될 것 같습니다.
>
> https://www.bioinf.jku.at/publications/older/2604.pdf
>
> 
>
> 논문의 내용이 방대하여 아래의 유튜브 영상을 참고하시는 것도 도움이 될 것 같습니다.
> 질문해주신 내용에 해당 하는 부분은 42분 10초에 해당합니다.
>
> https://www.youtube.com/watch?v=2ngo9-YCxzY



## 2. RNN에서 인퍼런스에 대한 질문입니다.

RNN 인퍼런스 과정에 대한 질문입니다.
<img src="https://cphinf.pstatic.net/mooc/20210216_292/1613474937301Gttut_PNG/mceclip0.png" alt="img" style="zoom: 50%;" />

여기 그림에서 학습이 완료된 다음에 "h"를 넣어서 예측한 값이 "e"로 나오게 되는데, softmax값을 확인하면 월등하게 "o"가 확률이 높습니다. cs231n 강의에서는 이 부분을 놓고 랜덤으로 샘플링해서 꺼낸다고 설명을 하셨는데 혹시 이 샘플링은 어떤 식으로 이뤄지는 것인지 궁금합니다! 



>RNN language model을 학습할 수 있는 방법은 아주 다양할 수 있습니다. 강의에서 소개드린 방법이 그 중 하나인 teacher forcing이 되겠고, 정답 character가 아닌 모델의 실제 output을 다음 time step의 input으로 넣어줄 수 있는 경우도 여러가지로 생각해볼 수 있습니다.
>
>가장 naive한 방법은 softmax output에 argmax 연산을 해서 가장 높은 확률 값을 갖는 character를 넘겨줄 수 있습니다. cs231n 강의에서는 이 방법 말고 softmax output을 확률 분포로 보고 여기서 sampling을 하는 방법을 말하는 것 같은데요 h: 3%, e: 13% l: 0%, o: 84%의 확률 분포에서 sampling을 하여 다음 time step의 input으로 넣어주고 학습을 할 수 있습니다. 이 경우 argmax를 하면 무조건 o가 넘겨지는 것과 달리 낮은 확률이지만 h나 e 문자도 다음 time step에 넘겨질 수 있습니다.
>
>더 나아가면 다양한 decoding strategy를 이용할 수 있는데 50%까지 해당하는 character 후보군에서만 sampling을 한다거나, 후보 character가 100개인 경우 확률값 기준 상위 5개의 character 중에서만 sampling을 하는 등의 방법이 있습니다.



## 3. RNN의 모델의 Many-to-many 출력 시퀀스 길이의 가변성에 대해 질문 드립니다.

RNN 모델의 경우 many-to-many 문제에 해당하는 machine translation이 가능한 것으로 배웠습니다. 다만 출력 시퀀스 길이에 대해 궁금한 점이 있습니다.

'Translation 모델'이 생성할 수 있는 출력 시퀀스의 최대 길이는 입력 시퀀스의 길이에 의존하나요? 다르게 말씀드리자면, 출력 시퀀스의 최대 차원 수는 입력 시퀀스의 차원 수보다 작아야 하나요? 또는 같아야 하나요?

복잡하거나 함축된 문장을 번역할 수록 번역 전 입력 시퀀스와 번역 후 출력 시퀀스의 길이가 다를 수도 있다고 생각하는데, 이러한 경우 시퀀스 길이가 어떻게 처리되는 것인지 궁금합니다.

가령, '식은 죽 먹기지!'의 국문이 'Easy peasy!'의 영문으로 번역되는 경우, 국문의 입력 시퀀스보다 영문의 출력 시퀀스의 길이(토큰 수)가 비교적 짧은 것으로 보이는데, Translation 모델에서는 이러한 번역 과정에서 시퀀스를 어떤 식으로 생성하여 최종적인 문장을 출력하는 것인가요?



> 좋은 질문인 것 같습니다.
> tokenization에 따라 달라지긴 하겠지만 언어 pair마다 한 문장을 기준으로 했을 때 평균 입출력 길이가 달라질 수 있습니다. 오늘 배운 RNN many-to-many의 경우에는 source와 target 문장의 max_sequence_length는 동일할 수밖에 없습니다.
>
> 해당 이슈를 해결할 수 있는 방법 중 하나가 내일 배우게 될 sequence to sequence 구조인데요. 금일 과제에서도 대략적으로 나와 있지만, encoder-decoder를 두어 encoder에서는 source sentence를 입력하고, decoder에서는 target sentence를 generate하게 해서 서로 다른 max_sequence_length를 가지도록 할 수 있습니다.
>
> 더 나아가서, 사전 설정된 max_sequence_length보다 더 긴 문장을 생성하고자 하는 것도 연구가 되고 있는데요 다음 논문을 참고하시면 좋을 것 같습니다. (Transformer-XL: Attentive Langauge Models Beyond a Fixed-Length Context, https://arxiv.org/pdf/1901.02860.pdf)



## 4. Character level language model 질문.

강의자료 20페이지에 보면 학습시에 매 time step마다 input으로 정답 문자들(h,e,l,l)을 넣어주고 있습니다.

근데 21페이지에 보면 테스트시에는 정답을 모르므로 매 time step의 예측결과를 다음 time step의 input으로 넣어주고 있습니다.

여기서 궁금한 점은 학습시에도 테스트에서 처럼 매 time step의 예측결과를 다음 time step의 input으로 넣어주어야 하는게 아닌가요? 학습시에는 마치 매 time step마다 정답을 알려 주고 있는 듯한 느낌입니다. 예를 들어 첫번쨰 output을 e가 아닌 a로 예측했다 해도 다음 time step의 input으로 e를 넣어줌으로써 원래 정답은 e 였다는걸 알려주고 있는 것 같습니다. 근데 테스트시에는 정답을 모르므로 이런식으로 될 수가 없는데 어째서 학습과 테스트를 다르게 해주고 있는건지 궁금합니다.



> 말씀하신 대로 학습시에도 예측 결과 그대로를 다음 time step에 넣어줄 수 있습니다. 이때 한가지 생각해볼만한 점은 이전 time step에서의 결과가 다음 time step의 출력에 영향을 미친다는 점입니다 (recursive algorithm). 그렇다면 학습 초기에 parameter들은 거의 random character/word를 출력하게 되고, 이는 다음 time step에서 제대로 된 character/word를 출력하는 것을 학습하기 어렵게 만듭니다. 잘못된 입력으로 인해 현재 또는 다음 hidden state 역시 영향을 받기 때문입니다. 이것이 누적되면서 학습이 어려워질 수 있습니다. 따라서, train phase에서는 무조건 정답 character/word를 다음 time step에 넣어주는 teacher forcing 기법을 적용하게 되는데요.
> 다음과 같은 방법으로 절충안(?)을 생각해볼 수 있습니다. 실제 학습과정을 테스트 환경과 유사하게 만들기 위해 초반 학습 과정에서 어떤 일정 iteration, perplexity, epoch, loss 등을 기준으로 정하고 해당 iteration, epoch 이후 또는 loss, perplexity 이하에서부터는 실제 모델의 출력 결과를 다음 time step에 넘겨주어 학습시킬 수 있겠습니다.





+그리고 teacher forcing이 regularization처럼 작동하는 건지, 실제로 teacher forcing을 사용하지 않고 학습하게 되면 앞 sequence에서는 틀리더라도 뒤쪽 sequence에서는 올바른 character/word를 출력할 수 있을 것 같은데 어떻게 생각하시나요?



> +저도 해당 technique의 정확한 motivation은 잘 모르겠습니다만, RNN으로 LM task를 학습하기 위한 technique으로 등장했고 ,
>
> (참고: http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.52.9724&rep=rep1&type=pdf) 
>
> 학습 phase에서 언제부터 teacher forcing을 사용하는지에 따라 적용하는 정도(?)를 조절하는 의미라면 강하게 할수록 overfitting이 일어난다고 볼 수 있습니다. L1, L2 regularization과 같은 기존 방법은 overfitting을 방지하는 용도로 사용하기 때문에 teacher forcing은 반대로 영향을 준다고 할 수 있을 것 같네요
>
> 실제로 기존 LM은 training corpus의 문장을 그대로 생성하는 경우를 보입니다. 자주 등장하지 않는 문장 패턴을 여러 epoch 동안 학습한다거나, 거의 유사한 문장 패턴이 많은 경우 이런 현상을 보이는 것 같은데요.
> 이런 문제를 해결하는 방향이 학습 방법을 개선하는 것 보다는 학습이 끝나고 generation할 때 stochastic inference를 하게 해서 해결하는 쪽으로 많이 연구되고 있는 것 같습니다. (참고: https://huggingface.co/blog/how-to-generate)
>
> 내일 배우게 될 beam search에서 비슷한 주제를 다루니 참고해보시면 좋을 것 같습니다. 
>
> 저는 teacher forcing을 전혀 사용하지 않고 학습시켜본 적이 없어서 실제 학습 결과가 어떨지는 잘모르겠는데요 ㅎㅎ 한번 실험해보시면 좋을 것 같습니다. 그리고 제가 말씀드렸던 절충안(?)과 유사한 방법이 다음 링크의 논문에서 제시된 것이 있어서 참고해보시면 좋을 것 같습니다. (Professor Forcing: A New Algorithm for Training Recurrent Networks, https://papers.nips.cc/paper/2016/file/16026d60ff9b54410b3435b403afd226-Paper.pdf)