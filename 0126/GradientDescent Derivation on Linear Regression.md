# 경사하강법으로 선형회귀 계수 구하기



- 목적식(오차의 L2 norm)을 최소로 하는 *`Beta`*를 구해야 한다.
- 이는 경사하강법 알고리즘을 통해서 이루어지는데 이 때, 다음과 같은 그레디언트 벡터의 계산이 필요하다.



![image-20210130195551581](GradientDescent Derivation on Linear Regression.assets/image-20210130195551581.png)



- 위 그레디언트 벡터에서 k번째 원소는 다음의 값을 구하는 것과 같으며, 이번 시간에는 이 결과를 직접 수식으로 유도해보며 과정을 더 자세히 이해하도록 해본다.

$$
\part\beta_k\lVert y-XB \rVert_2\
$$


$$
\begin{align}


\part\beta_k\lVert y-XB \rVert_2\ \ \ 


&= \part\beta_k[\frac{1}{n}{\sum_{i=1}^n(y_i-\sum_{j=1}^dX_{ij}\beta_j)^2} ]^{1/2}\\
\\


&= \frac{1}{2\sqrt{n}}[{\sum_{i=1}^n(y_i-\sum_{j=1}^dX_{ij}\beta_j)^2} ]^{-1/2}\ \times\ 

{\sum_{i=1}^n\part\beta_k(y_i-\sum_{j=1}^dX_{ij}\beta_j)^2}\\
\\


&= \frac{1}{2\sqrt{n}}[{\sum_{i=1}^n(y_i-\sum_{j=1}^dX_{ij}\beta_j)^2} ]^{-1/2} \times\ 

{\sum_{i=1}^n2(y_i-\sum_{j=1}^dX_{ij}\beta_j)\ \ \part\beta_k(y_i-\sum_{j=1}^dX_{ij}\beta_j)}\\
\\

&= \frac{1}{\sqrt{n}}[{\sum_{i=1}^n(y_i-\sum_{j=1}^dX_{ij}\beta_j)^2} ]^{-1/2} \times\ 

{\sum_{i=1}^n(y_i-\sum_{j=1}^dX_{ij}\beta_j)}\ (-X_{\cdot k})\\
\\

&= \frac
{{\sum_{i=1}^n(y_i-\sum_{j=1}^dX_{ij}\beta_j)}\ (-X_{\cdot k})}

{\sqrt{n}[{\sum_{i=1}^n(y_i-\sum_{j=1}^dX_{ij}\beta_j)^2} ]^{1/2}}\\\\


&= -\frac
{X_{\cdot k}^T(y-X\beta)}
{\sqrt{n}\sqrt{n}[\frac{1}{n}{\sum_{i=1}^n(y_i-\sum_{j=1}^dX_{ij}\beta_j)^2} ]^{1/2}}\\


&= -\frac
{X_{\cdot k}^T(y-X\beta)}
{n\lVert y-XB_k \rVert_2}


\end{align}
$$


- 따라서, 최종적으로 우리가 구하고자 하는 그레디언트 벡터는 다음과 같게 된다.

$$
\begin{align}

\nabla\beta_\lVert y-XB \rVert_2\ &= (\part\beta_1\lVert y-XB \rVert_2, ... , \part\beta_d\lVert y-XB \rVert_2)\\



&= (-\frac
{X_{\cdot 1}^T(y-X\beta)}
{n\lVert y-XB_k \rVert_2},
...
,
-\frac
{X_{\cdot d}^T(y-X\beta)}
{n\lVert y-XB_k \rVert_2})\\


&=-\frac
{X^T(y-X\beta)}
{n\lVert y-XB \rVert_2}


\end{align}
$$

- XB를 계수 B에 대해 미분한 결과인 X

$$
\beta^{(t+1)}\ ← \ \beta^{(t)}\ -\ \nabla_\beta\lVert{y-X\beta_{(t)}}\rVert
$$









