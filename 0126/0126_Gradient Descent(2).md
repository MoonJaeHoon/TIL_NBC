# 경사하강법 응용



<img src="../../../../../AppData/Roaming/Typora/typora-user-images/image-20210126124836357.png" alt="image-20210126124836357" style="zoom:80%;" />





![image-20210126125415158](../../../../../AppData/Roaming/Typora/typora-user-images/image-20210126125415158.png)



![image-20210126125428692](../../../../../AppData/Roaming/Typora/typora-user-images/image-20210126125428692.png)



# SGD가 뭔데?



![image-20210126131115905](../../../../../AppData/Roaming/Typora/typora-user-images/image-20210126131115905.png)





![image-20210126131133321](../../../../../AppData/Roaming/Typora/typora-user-images/image-20210126131133321.png)



## SGD의 원리



- Gradient Descent는 전체데이터를 가지고 목적식의 그레디언트 벡터를 계산합니다.
- SGD는 미니배치를 가지고 그레디언트 벡터를 계산합니다.



![image-20210126131528150](../../../../../AppData/Roaming/Typora/typora-user-images/image-20210126131528150.png)



- SGD는 미니배치를 가지고 그레디언트를 계산하는데, 미니배치는 확률적으로 선택되므로 목적식 모양이 바뀌게 됩니다.



![image-20210126131623328](../../../../../AppData/Roaming/Typora/typora-user-images/image-20210126131623328.png)



- 목적식의 모양이 바뀐다는 것은, 서로 다른 미니배치를 사용하여 곡선의 모양이 바뀐다는 것이다.
- 만약, Non-Convex인 경우에 Local Point에(극소점이나 극대점에) 도착해버린 경우에도 SGD는 확률적으로 목적식이 바뀌게 되기 때문에 해당 지점이 더이상 Local Point가 아니게 할 수 있는 확률이 있다.
- 따라서, (본래 Gradient Descent에서는 local point에 빠져 탈출이 불가능한 경우였다 하더라도)  SGD는 탈출이 가능하다.
- 또한, 전체데이터가 아니라 일부분 (미니배치)만을 가지고 그레디언트 벡터를 계산하기 때문에 각 화살표를 진행하는데 걸리는 시간이 훨씬 적게 걸린다 (같은 시간 대비 더 많이 움직일 수 있다.)



> 이러한 목적식이 계속 바뀌는 SGD의 특성상, 수렴하는 모습은 다음과 같다.

<img src="../../../../../AppData/Roaming/Typora/typora-user-images/image-20210126132702963.png" alt="image-20210126132702963" style="zoom:150%;" />





<img src="../../../../../AppData/Roaming/Typora/typora-user-images/image-20210126132723952.png" alt="image-20210126132723952" style="zoom:150%;" />



<img src="../../../../../AppData/Roaming/Typora/typora-user-images/image-20210126132738803.png" alt="image-20210126132738803" style="zoom:150%;" />

- 주의할 점 : SGD에서 미니배치 사이즈를 너무 작게 잡게 되면 GD보다 수렴속도가 느려질 수 있다.



