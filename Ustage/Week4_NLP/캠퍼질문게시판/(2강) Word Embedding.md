# 캠퍼 질문 게시판

## 1. Word Embedding의 학습이 잘 이해가 안돼요

강의 약 5분 40초쯤 부터 진행되는 내용인데요.

학습쌍에서 (study, I), (study, math) 가 있는데

study를 인풋으로 넣었을때 I 가 나올 확률도 높여주어야하고, math가 나올 확률도 높여줘야 한다면,

SGD로 학습하는 경우에 수렴가능한 학습인가요?

제 생각에는 가중치가 수렴하지 않고 진동할것 같아서요.



![img]((2강) Word Embedding.assets/mceclip0.png)



> **박기훈 조교님**
>
> 위의 (study,I) (study,math)의 경우 전체 학습 데이터셋의 극히 일부분에 불과하고, 실제로 여러 문장들에 대해 window sliding 방식을 통해 학습을 진행하게 되면 study에 대해 I, math외에도 school, pencil, book 등의 다양한 단어들이 등장할 수 있습니다.
> 그리고 I와 math에 대해서도 다른 문장에서 study 외의 단어들이 등장할 수 있습니다. 이는 결국 각각의 단어에 대해 서로 다른 빈도의 다른 단어들을 등장시킴으로써 단어마다 적절한 embedding 값을 가질 수 있도록 합니다 !



+ 추가질문

다른 데이터셋이 많아진다고 해도 결국 학습과정에서

study가 math를 뱉어내도록 학습이 진행되는 과정이 있을테고

stydy가 school을 뱉어내도록 학습하는 부분이 있을것 같은데요.

 

어떤 데이터셋에서 study와 함께 등장한 math의 비율이 0.3 school의 비율이 0.3 그외 0.4라고 했을때

(study, math)쌍을 학습할때는 출력에서 math에 해당하는 부분만 1을 출력하고,

(study, school)쌍을 학습할때는 school에 해당하는 부분만 1을 출력해야 하는데

이 과정에서 한 점으로 수렴하는게 어떻게 가능한가요?

아니면 말씀하신 적절한 embedding 값이라는게 한 점으로 수렴하진 않지만 learning_rate를 작은 값으로 주면 진동폭을 작게 만들어서 쓸만한 값을 가지게 한다는 뜻인가요?



>**김태희 조교님**
>
>말씀하신대로 꼭 word2vec 모델이 주어진 dataset/corpus에 대해 수렴한다는 보장은 없습니다. word vector size, learning rate, learning rate scheduler 등에 의해 해당 task(window size에 등장하는 단어 맞추기)에 대해 수렴할 수도 있고 아닐 수도 있습니다. task의 특성상 loss가 0이 될 수 없겠지만, model capacity에 의해 일정 iteration 또는 epoch 동안 loss가 떨어지지 않을 때까지 학습을 하고 해당 word embedding을 가져다가 쓸 수 있습니다.
>실제로는 gensim등에서 미리 학습된 word embedding을 가져다가 쓰는 경우가 대부분이고, 직접 word embedding을 학습하는 경우 많은 epoch으로 학습할수록 target task에 대한 성능이 증가한다는 이야기가 많습니다. word embedding 학습에 쓰이는 corpus의 크기가 매우 크기 때문에 많은 분들이 그저 가능한 많은 epoch를 돌리고 수렴이 되지 않은 상태에서 학습을 중단하고 해당 word embedding을 쓰는 것 같습니다.