# 경사하강법

## 1. Introduction

> **미분의 정의** 
> $$
> f'(x) = lim_{h→0}\frac{f(x+h)-f(x)}{h}
> $$

ex)

> $$
> f(x) = x^2+2x+3
> $$

> $$
> f'(x) = 2x+2
> $$



```python
import sympy as sym
from sympy.abc import x

sym.diff(sym.poly(x**2+2*x+3),x)

```

```
Poly(2*x + 2, x, domain='ZZ')
```



## 2. 미분을 어디에 쓰는가?



### 1) 경사상승법(gradient ascent)

- 접선의 기울기를 사용하여 어느 방향으로 점을 움직여야 함수값이 증가할까?
- 양수인 경우와 음수인 경우 모두 `x+f(x)`를 취해주면 된다.
- 이처럼 미분값을 더하는 것을 경사상승법이라 하며 함수의 극대값의 위치를 구할 때 사용한다.
- 목표함수를 최대화시킬 때 사용

![image-20210126102302517](../../../../AppData/Roaming/Typora/typora-user-images/image-20210126102302517.png)



![image-20210126101533592](../../../../AppData/Roaming/Typora/typora-user-images/image-20210126101533592.png)



### 2) 경사하강법(gradient descent)

- 접선의 기울기를 사용하여 어느 방향으로 점을 움직여야 함수값이 감소할까?
- 양수인 경우와 음수인 경우 모두 `x-f(x)`를 취해주면 된다.
- 이처럼 미분값을 빼는 것을 경사하강법이라 하며 함수의 극소값의 위치를 구할 때 사용한다.
- 목표함수를 최소화시킬 때

![image-20210126102237670](../../../../AppData/Roaming/Typora/typora-user-images/image-20210126102237670.png)



![image-20210126101451401](../../../../AppData/Roaming/Typora/typora-user-images/image-20210126101451401.png)



## 3. 변수가 벡터일 때의 미분

- 벡터가 입력인 다변수 함수의 경우, 편미분(partial differentiation)을 사용한다.



> $$
> {\partial}_{x_i}
> f(x) = lim_{h→0}\frac{f(x+he_i)-f(x)}{h}
> $$

ex)

> $$
> f(x,y) = x^2+2xy+3 + cos(x+2y)
> $$

> $$
> f'(x) = 2x+2y-sin(x+2y)
> $$



```python
import sympy as sym
from sympy.abc import x,y

sym.diff(sym.poly(x**2+2*x*y+3)+sym.cos(x+2*y),x)
# x에 대한 편미분

```

```
2*x + 2*y - sin(x + 2*y)
```



- 각 변수 별로 편미분을 계산한 그레디언트 벡터를 이용하여 경사하강/경사상승법에 사용하는 것이다.

$$
{\nabla}f = ({\partial}_{x_1}f,{\partial}_{x_2}f, ... ,{\partial}_{x_d}f )
$$

$$
where\ \ {\partial}_{x_i}f \ = \ \ lim_{h→0}\frac{f(x+he_i)-f(x)}{h}
$$

- 기하학적 의미

$$
-{\nabla}f는\ 각\ 점에서\ 가장\ 빨리\ 감소하게\ 되는\\ 방향을\ 나타내는\ 것입니다.
$$



<img src="../../../../AppData/Roaming/Typora/typora-user-images/image-20210126115114135.png" alt="image-20210126115114135" style="zoom:80%;" />





