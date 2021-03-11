# 경사하강법 기본

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


- 미분해주는 파이썬의 라이브러리도 있음 (`sym`)
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

![image-20210130233116775](0126_Gradient Descent(1).assets/image-20210130233116775.png)



![image-20210130233100891](0126_Gradient Descent(1).assets/image-20210130233100891.png)



### 2) 경사하강법(gradient descent)

- 접선의 기울기를 사용하여 어느 방향으로 점을 움직여야 함수값이 감소할까?
- 양수인 경우와 음수인 경우 모두 `x-f(x)`를 취해주면 된다.
- 이처럼 미분값을 빼는 것을 경사하강법이라 하며 함수의 극소값의 위치를 구할 때 사용한다.
- 목표함수를 최소화시킬 때

![image-20210130233141186](0126_Gradient Descent(1).assets/image-20210130233141186.png)

![image-20210130233159741](0126_Gradient Descent(1).assets/image-20210130233159741.png)

## 3. 변수가 벡터일 때의 미분

- 다차원의 변수가 입력으로 들어오는 경우 이동할 때 굉장히 많은 방향으로 설정을 해볼 수가 있다.
- 따라서 이렇게 벡터가 입력인 다변수 함수의 경우, `편미분(partial differentiation)`을 사용한다.

> **편도함수의 정의**

> $$
> {\partial}_{x_i}
> f(x) = lim_{h→0}\frac{f(x+he_i)-f(x)}{h}\\\\
> where\ e_i = (0,0,...1,...0) (1\ \ \ is\ \ \ i-th\ \ element)
> $$

ex)



> $$
> f(x,y) = x^2+2xy+3 + cos(x+2y)
> $$

> $$
> {\partial}_{x_i}
> f(x)=f'(x) = 2x+2y-sin(x+2y)
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



- 수식적으로는 각 변수 별로 편미분을 계산한 **그레디언트 벡터**를 이용하여 `경사하강/경사상승법`에 사용합니다.
- 이렇게 앞서 사용했던 미분값인 `f'(x)` 대신 **그레디언트 벡터**를 사용하면, `변수 x=(x1,...xd)`를 동시에 업데이트가 가능하게 되는 것입니다.

$$
{\nabla}f = ({\partial}_{x_1}f,{\partial}_{x_2}f, ... ,{\partial}_{x_d}f )
$$

$$
where\ \ {\partial}_{x_i}f \ = \ \ lim_{h→0}\frac{f(x+he_i)-f(x)}{h}
$$



**기하학적 의미**

- (x,y,z)공간의 그래프가 있다고 했을 때, 해당 함수의 그레디언트 벡터에 음수를 취해준 값은 그림과 같이 아래(극솟값)으로 향하는 화살표로서 그 역할을 하게 된다.
- 이 화살표만 따라가게 되면 (마이너스 그레디언트 벡터 방향 따라가면) 극솟값에 도달할 수 있게 되는 것이 그 원리이다. 

![image-20210130233254500](0126_Gradient Descent(1).assets/image-20210130233254500.png)





$$
결국,\ 그레디언트 벡터 \ -{\nabla}f는\ \ 각\ \ 점에서\ 가장\ 빨리\ 감소하게\ 되는\\ 방향을\ 나타내는\ 것입니다.
$$



![image-20210130233309922](0126_Gradient Descent(1).assets/image-20210130233309922.png)



