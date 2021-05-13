## 1. **LSTM (Long Short Term Memory)**



**`Long Short-Term Memory(LSTM)`**은 기존 RNN모델에서 **`Vanishing/Exploding Gradient`** 문제를 해결하고, **`long-term depandancy`** 문제를 개선한 모델이다.

hidden state를 단기(short-term)기억소자로 볼 수 있고, 보다 먼 timestep(long-term)의 정보까지 잘 반영하도록 만들었기 때문에 이러한 이름이 붙었다.

<br>

### 구조

기존의 RNN 모델이 다음과 같은 형태였다면,
$$
h _
t
​	
 =f _
w
​	
 (x _
t
​	
 ,h _
{t−1}
​	
 )
$$
LSTM 모델의 형태는 다음과 같다.
$$
\{C _
t
​	
 ,h _
t
​	
 \}=LSTM(x _
t
​	
 ,C_ 
{t−1}
​	
 ,h _
{t−1}
​	
 )
$$

$$
C_t
​	
  : Cell\ State\ Vector
$$

$$
h _
t
​	
  : Hidden State Vector
$$



LSTM은 hidden state 외에 Cell state를 추가로 둠으로써, Vanilla RNN에서 sequence 길이가 길어지면 학습이 안 되는 문제를 해결하기 위해 만들어진 구조입니다.

![image-20210316024818758](Lecture4_LSTM%20and%20GRU.assets/img1.png)



LSTM 구조를 더 자세히 살펴보면 아래와 같습니다.

![img](Lecture4_LSTM%20and%20GRU.assets/img2.png)

 

**previous cell state**는 네트워크 내부에서만 흘러가고, **11부터 t−1t−1의 정보를 다 취합해서 요약해준 정보**를 뜻합니다.

**previous hidden state**는 **나가는 output정보**입니다.

따라서 실제로 나가는 값(출력되는 값)은 output(hidden state)밖에 없습니다.

 

### **1) Forget Gate : 어떤 정보를 버릴지**

 



![img](Lecture4_LSTM%20and%20GRU.assets/img3.png)

 

여기서 **"[ ]"**가 **concat 하는** 것을 뜻합니다.

현재 입력 xtxt와 이전의 output ht−1ht−1이 들어가서 ftft라는 숫자(값)를 얻어내게 됩니다.

위 식은 sigmoid σσ를 통과하기 때문에 ftft는 항상 0~1사이의 값입니다.

 

ftft는 cell state에서 나오는 정보 중에 **어떤 값을 버리고 살릴지를 결정하는 역할**을 합니다.

즉, cell state를 버리고 살릴지 결정하는 것은 현재 입력 xtxt와 이전의 output ht−1ht−1을 통해 결정이 됩니다.

 

### **2) Input Gate : 어떤 정보를 cell state에 추가할지**

 



![img](Lecture4_LSTM%20and%20GRU.assets/img4.png)

 

itit는 **어떤 정보를 추가할지 말지를** 현재 입력 xtxt와 이전의 output ht−1ht−1을 사용해서 결정합니다.

~ctc~t는 **추가할 정보의 내용**을 뜻하며, 현재 입력 ht−1,xtht−1,xt을 "따로 학습되는 neural network" (가중치가 WCWC)에 집어넣고 tanhtanh (정규화)을 취해준 값입니다.

 

### **3) Update Gate : cell state 업데이트**

 



![img](Lecture4_LSTM%20and%20GRU.assets/img5.png)

 

Update Gate는 **previous cell state와 CtCt를 잘 조합해서 새로운 cell state로 update 하는** 과정입니다.

 

Ct(새로운cellstate)=ft∗Ct−1(버릴것은버리고)+it∗~Ct(새롭게추가)Ct(새로운cellstate)=ft∗Ct−1(버릴것은버리고)+it∗C~t(새롭게추가)

 

### **4) Output Gate : 어떤 값을 내보낼지**

 



![img](Lecture4_LSTM%20and%20GRU.assets/img6.png)

 

otot는 **어떤 값을 밖으로 내보낼지를 결정**합니다.

otot를 사용해서 htht (output)을 내보내고, 다음 previous hidden state로 보냅니다.

 

 

## 2. **GRU (Gated Recurrent Unit)**

 

GRU는 **LSTM에서 parameter를 줄인 모델**입니다.

LSTM보다 GRU 성능이 더 좋다는 실험적인 결과가 있는데요. 그 이유는 parameter가 적기 때문에 generlization performance가 올라가기 때문입니다.

 



![img](Lecture4_LSTM%20and%20GRU.assets/img7.png)

 

GRU는 **reset gate**와 **update gate,** 2개의 gate를 사용합니다.

GRU의 특징은 **cell state가 없고 hidden state가 곧 output이며, 다음 순서의 previous hidden state**를 나타냅니다.

또한, **output gate가 없습니다.**



