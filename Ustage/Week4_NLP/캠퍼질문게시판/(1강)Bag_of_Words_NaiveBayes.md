# 캠퍼 질문 게시판

## 1. Naive Bayes classifier 계산에서 질문 드립니다!

Naive Bayes classifier에서 확률 계산을 할 때 각 단어가 등장을 모두 독립으로 가정하고 계산을 해주게 됩니다. 혹시 항상 독립으로 가정하고 계산을 해주는지 경우에 따라서 다른 계산법을 사용하는지 궁금합니다.



> 말씀하신대로 Naive Bayes classifier는 각각의 단어들이 등장할 확률을 모두 독립으로 보고 계산을 하는 것이 맞습니다. 그리고 항상 독립으로 가정하고 계산을 해주는지에 대해서 질문 주셨는데, 적어도 Naive Bayes Classifier의 정의상으로는 그렇습니다. 애초에 "naive", 즉 단순하게 그냥 모든 feature의 등장을 각각 독립으로 보는 것이 Naive Bayes classifier의 기본 전제이므로 특별히 다른 방식으로 계산해주는 방식은 적어도 이 알고리즘 상에선 없다고 보셔도 무방합니다.

> 덧붙여 당연히 실제 케이스에선 반드시 모든 단어들이 독립적이라고 볼 수는 없고 그래서 이제 딥러닝을 비롯한 조금 더 고성능을 내는 방법들이 현대에 와서 제시가 되었습니다. 다만 Naive Bayes로도 그래도 꽤 괜찮은 성능이 나오는 경우가 아직도 있기 때문에 아주 단순하기만 한 가정이라고는 볼 수 없을지도 모르겠습니다.

