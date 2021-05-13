## 1. Self-Attention 개념 중 k,q,v의 차원과 seq_len에 관해 질문드립니다.

안녕하세요 조교님! Self-Attention 개념 중 궁금한 내용이 있어 질문드립니다.



1. 강의자료 8p 그림에서 Key matrix의 행의 크기가 seq_len와 달라도 되는 이유가 무엇일까요? |K|와 |V|의 값이 4로, 시퀀스 길이 3보다 더 클 수 있다는 부분을 이해하지 못하겠습니다. 예를 들어 'I go home'이라는 문장이 있을 때 'l', 'go', 'home' 단어 중 특정 단어를 중복으로 사용해 key vector를 생성한다는 의미일까요?

 <img src="https://cphinf.pstatic.net/mooc/20210219_26/1613676454874yTpYH_PNG/mceclip2.png" alt="img" style="zoom:67%;" />



2. 강의 내용 중에 d_q와 d_k는 같아야 하지만, d_v는 둘과 다른 값이어도 된다는 내용이 있었습니다.

(d_q: Query vector dimension, d_k: Key vector dimension, d_v: Value vector dimension, d_model: transformer의 모든 층의 출력차원으로 통일하려는 값)

<img src="https://cphinf.pstatic.net/mooc/20210219_261/1613676671938Wgxx7_PNG/mceclip0.png" alt="img" style="zoom:67%;" />

d_v이 어떤 값이건 Value vector들이 가중 평균된 값으로 seq_len 차원을 가진 인코딩 벡터를 형성하기 때문이라고 알고 있습니다. 제가 이해한 내용이 맞을까요? 그런데 다른 참고 자료(https://wikidocs.net/31379)에서는 d_q==d_k==d_v==(d_model/num_heads) 라는 가정이 고정되어 있더라구요. 그래서 "attention head의 shape이 (seq_len,d_v)일 때 attention head들을 모두 concat한 결과가 (seq_len,d_v*num_heads)==(seq_len,d_model)으로 입력층 행렬 크기가 그대로 유지될 수 있었다" 라는 내용이 나옵니다.

만약 d_v가 d_q와 d_k와 같지 않다면 Multi-head attention matrix의 차원이 d_model이 되지 못할수도 있는 것 아닌가요? 제가 어떤 내용을 잘못 이해하고 있는지 몰라서 도움 요청드립니다.



> 1, 2번 모두 연관되는 내용이므로 한 번에 설명을 드리면 우선 Query, Key, Value의 개념을 다시 짚고 넘어갈 필요가 있습니다.
>
> Python에서 dict라는 자료구조를 기억하실 겁니다. Key와 value가 서로 대응이 되어 있고 key를 통해 value를 찾아낼 수 있는 구조로 되어 있죠. 기본적으로 key는 각각 자신과 대응되는 value를 가지고 있으므로 key와 value는 같은 개수로 존재하는 pair, 세트라고 보시면 됩니다. 그리고 query는 내가 실제로 찾고자 하는, 즉 내가 직접 질의하고자 하는 key가 됩니다. 예를 들어, {'a': 1, 'b': 2, 'c': 3}라는 dict가 있으면 (key, value)는 각각 ('a', 1), ('b', 2), ('c', 3)이 있는 것이고 이제 이 dict한테 내가 'b'의 value 좀 찾아달라고 요청을 하면 이 'b'가 바로 제가 질의하고자 하는 query가 되는 것이죠. 따라서 query는 key를 질의해야 하는 것이므로 query와 key는 같은 형식 또는 같은 선 상에 있는 정보여야 가능합니다.
>
>  
>
> Attention에서 말하는 q, k, v도 결국 위의 query, key, value와 같습니다. 다만 이 값들이 전부 벡터일 뿐인 것이고, query가 key를 찾는 과정이 단순히 있는지 없는지 찾는 것이 아닌 벡터 간 내적일 뿐인 거죠. (query 입장에선 key들 중 그나마 자신과 유사한 key를 찾는 것임.)
>
> query와 key가 같은 차원이어야 하는 이유가 이렇게 찾는 과정을 수행하기 위해선 이들의 형식이 같은 상태여야 하기 때문임.
>
> 내가 현재 집중하고자 하는 query 벡터가 q라면 이것을 대상이 되는 모든 key 벡터인 k들과 각각 내적을 하고 그렇게 얻은 내적 값, 즉 유사도 값을 구하여 그 유사도 값에 따라 선택적으로 각 k가 pair로 가지고 있는 value인 v들을 선택적으로 가져오는 것입니다. 그것에 이제 가중합(평균)을 내는 거죠.
>
>  
>
> 기본적으로 transformer의 multi-head attention에선 어차피 동일한 문장이 각각 q, k, v로 변환된 것이기 때문에 사실 seq_len과 d_model, 또는 d_k와 같은 차원이 모두 동일합니다. 따라서 위의 위키 독스의 말이 틀린 것은 아닙니다. 그럼 왜 강의에선 달라도 된다고 설명이 된 거냐면 실제로 위의 dict 예시처럼 pair를 이루는 값이나 형식만 같아도 multi-head attention은 성립하기 때문입니다.
>
> 즉, 일반적으로 확장해서 보면 어차피 q는 k랑만 내적을 하기 때문에 q랑 k의 차원 수만 같아도 내적 계산이 가능하고 따라서 v의 차원과는 달라도 상관이 없고(2번 질문의 답) 한편 q의 개수와는 관계없이 k랑 v는 형식, 즉 차원이 달라도 개수는 무조건 같아야 하는 pair 관계이므로 각 token의 개수인 seq_len이 동일해야 하는 것이기 때문에 k와 v의 길이만 같다면 q와 k의 길이는 달라도 무방합니다.(1번 질문의 답)
>
> 용어정리 예시)
>
> - seq = [I, go, home], seq_len = 3
>
> - query1 (token1) : I = [1,0,0]
>
> - query2 (token2) : go = [0,1,0]
>
> - query3 (token3) : home = [0,0,1]
>
> - k1,k2,k3와 v1,v2,v3도 똑같이 I와 go와 home으로부터 나온 출처가 똑같은 벡터들임.
>
>  
>
> 그리고 실습 8강에서도 설명을 하지만 실제로 1번 질문처럼 q와 k의 길이가 서로 다른 케이스를 소개했는데, 바로 decoder 안에 있는 encoder-decoder 간 attention을 하는 경우입니다. Encoder에 들어가는 src sentence의 길이와 Decoder에 들어가는 trg sentence의 길이를 물론 같게 맞춰줄 수는 있지만 다를 경우도 있습니다. 이런 경우도 실습 코드에서 보시면 아시겠지만 결국 attention matrix의 row와 column 개수만 다를 뿐 계산이 가능한 것을 보실 수 있습니다. 이 경우가 바로 1번에서 나온, q의 길이(decoder input의 token 개수)와 k의 길이(encoder input의 token 개수)가 다른 경우이고 이 경우에도 k와 v의 길이는 동일하게 유지되는 것을 볼 수 있습니다.(왜냐하면 같은 encoder input의 길이이므로) 동시에 내적이 이뤄지는 q와 k의 차원 역시 같죠. (왜냐하면 query는 key와 동일한 형식이어야 하므로)
>
>  
>
> 답변이 좀 길어졌는데 궁금하신 점 있으면 추가로 말씀해주시기 바랍니다. 감사합니다.



```
정리해보자면,
q,k의 차원수는 같아야 하고 v는 차원수가 달라도 됨.
q라는 녀석은 본인과 유사한 k를 찾기 위한 역할일 뿐이기 때문에 q가 몇개 있든 상관없음
대신에 k와 v는 dict 구조처럼 대응되는 구조이기 때문에 그 개수가 같아야 함. (seq_len)

하지만 결국 여기 Transformer의 Multi-head-Attention 메커니즘에선 q,k,v가 같은 어떤 seq로부터 나온 것이기 때문에 dimension이건 seq_len이건 다 똑같음 ㅎㅎ

강의내용에서 저렇게 다를 수 있다는 것을 언급하고 그림으로 보여준 것은 개념적으로 확실히 짚고 넘어가라는 의미였음.
Multi-head-Attention 메커니즘을 충분히 저렇게도 다르게 쓸 수 있다는 것이지.
다르게 쓰고 있는 예시가 바로 q,k,v가 동일한 어떤 seq로부터가 아니라 다른 seq로부터 나온 것일 때이다.
바로 그것이 Decoder 中 encoder-decoder 간 attention을 하는 경우.
```

