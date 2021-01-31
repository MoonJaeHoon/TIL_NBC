# [Pandas] Series와 DataFrame 연산

## 0. Series의 기본 연산

```python
import pandas as pd
from pandas import Series
from pandas import DataFrame

import numpy as np
```


```python
dict_data = {"a": 1, "b": 2, "c": 3, "d": 4, "e": 5}
example_obj = Series(dict_data, dtype=np.float32, name="example_data")
example_obj
```


    a    1.0
    b    2.0
    c    3.0
    d    4.0
    e    5.0
    Name: example_data, dtype: float32



- `Series`에서는 `astype()` 메소드를 통해 형변환을 이뤄낼 수가 있다.


```python
example_obj = example_obj.astype(float)
example_obj["a"] = 3.2
example_obj
```


    a    3.2
    b    2.0
    c    3.0
    d    4.0
    e    5.0
    Name: example_data, dtype: float64



- `Series`에 대소비교 연산자를 적용하면 `Boolean value`들을 가지는 `Series`가 된다.
- 이외에도 다양한 연산자를 적용할 수가 있다.

- 그리고 `Boolean` `sequence`를 통해 다음과 같이 `Selection`(filtering)도 가능하다.

```python
example_obj[example_obj > 2]
```

```
a    3.2
c    3.0
d    4.0
e    5.0
Name: example_data, dtype: float64
```



- 다음과 같이 다양한 연산자들이 `Series`에 적용가능


```python
example_obj * 2
```


    a     6.4
    b     4.0
    c     6.0
    d     8.0
    e    10.0
    Name: example_data, dtype: float64




```python
np.exp(example_obj)  # np.abs , np.log 등
```


    alphabet
    a     24.532530
    b      7.389056
    c     20.085537
    d     54.598150
    e    148.413159
    Name: number, dtype: float64



- `f` 인덱스에 `'가'` value 추가


```python
example_obj['f']='가'
example_obj
```


    alphabet
    a    3.2
    b    2.0
    c    3.0
    d    4.0
    e    5.0
    f      가
    Name: number, dtype: object




```python
"b" in example_obj    # 'b' in example_obj.index
```


    True




```python
'가' in example_obj	# False (인덱스에 대하여 찾기 때문)
```


    False




```python
'가' in example_obj.values	# True (Value에 대해 찾았다.)
```


    True





### 1. Series 간의 연산

> 무조건 인덱스를 기준으로 연산을 한다고 생각하면 된다.




```python
s1 = Series(range(1, 6), index=list("abced"))
s1
```


    a    1
    b    2
    c    3
    e    4
    d    5
    dtype: int64




```python
s2 = Series(range(5, 11), index=list("bcedef"))
s2
```


    b     5
    c     6
    e     7
    d     8
    e     9
    f    10
    dtype: int64




```python
s1 + s2
```


    a     NaN
    b     7.0
    c     9.0
    d    13.0
    e    11.0
    e    13.0
    f     NaN
    dtype: float64

- `s1`에는 `f`인덱스, `s2`에는 `a`인덱스에 값이 없기 때문에 `NaN(Not a Number)`을 반환한다.




```python
s1.add(s2)
```


    a     NaN
    b     7.0
    c     9.0
    d    13.0
    e    11.0
    e    13.0
    f     NaN
    dtype: float64

- `+` 연산자 대신 `.add()` 메소드를 사용할 수도 있다.



> 다음 예시에서는 하나의 `Series` 안에 중복된 `index`값을 갖는 경우이다.
>
> `examp1` 객체의 `'e'` 인덱스가 2개 중복되어 있고 각각 다른 `value`를 가지고 있다.
>
> 이 경우 어떻게 연산이 이루어질까?




```python
examp1 = pd.Series(data = np.arange(10), index=list('abcdefgeij'))
print(examp1)

examp2 = pd.Series(data = np.arange(5), index=list('acefg'))
print(examp2)
```

    a    0
    b    1
    c    2
    d    3
    e    4
    f    5
    g    6
    e    7
    i    8
    j    9
    dtype: int32
    a    0
    c    1
    e    2
    f    3
    g    4
    dtype: int32



- 우선, `examp1`에는 있지만 `examp2`에는 없는 인덱스들에 대하여 `NaN`을 반환하였다
- `examp1`이 가지는 2개의 `'e'` index에 대하여서는, 각각에 대하여 `examp2`의 `'e'`인덱스 해당 value 값을 더해주었다.

```python
examp2+examp1 # 2개 중 어느 한 시리즈의 같은 인덱스를 가지는 value가 두개일 경우, 각각에 대해서 두번 더해줌.
```


    a     0.0
    b     NaN
    c     3.0
    d     NaN
    e     6.0
    e     9.0
    f     8.0
    g    10.0
    i     NaN
    j     NaN
    dtype: float64



- 위와 같은 결과이다. `Series` 연산에서 순서는 상관이 없다.


```python
examp1+examp2 # 2개 중 어느 한 시리즈의 같은 인덱스를 가지는 value가 두개일 경우, 각각에 대해서 두번 더해줌.
```


    a     0.0
    b     NaN
    c     3.0
    d     NaN
    e     6.0
    e     9.0
    f     8.0
    g    10.0
    i     NaN
    j     NaN
    dtype: float64



> 추가적으로, 그럼 중복되는 index를 한쪽만 가진 게 아니라 피연산 `Series` 둘 다 가지고 있다면, 연산시 어떤 결과가 나오게 될까?
>
> 다음 예시를 보자.



- `examp3` `'e'` index값이 2개
- `exap4`도  `'e'` index값이 2개


```python
examp3 = pd.Series(data = np.arange(10), index=list('abcdefgeij'))
print(examp3)

examp4 = pd.Series(data = np.arange(5), index=list('acefe'))
print(examp4)
```

    a    0
    b    1
    c    2
    d    3
    e    4
    f    5
    g    6
    e    7
    i    8
    j    9
    dtype: int32
    a    0
    c    1
    e    2
    f    3
    e    4
    dtype: int32



>  결과 
>
> - 모든 경우의 수 (2X2)에 대하여 연산을 진행하고, 결과값들로서 반환해줍니다.
> - `e` index에 해당하는 `value`가 총 4개 생겼습니다.

```python
examp3+examp4    # 2개의 시리즈 모두 각각 같은 인덱스(e)에 2개씩 값을 가지면 모든 경우의 수에 대해 계산하여 결과는 인덱스 e에 대하여 2*2=4개의 value가 생김
```


    a     0.0
    b     NaN
    c     3.0
    d     NaN
    e     6.0
    e     8.0
    e     9.0
    e    11.0
    f     8.0
    g     NaN
    i     NaN
    j     NaN
    dtype: float64







### 2. DataFrame 간의 연산

> 데이터프레임 역시 인덱스를 기준으로 연산을 진행하는데, `row_index`, `column_index`를 각각 모두 고려해서 연산을 진행한다.


```python
df1 = DataFrame(np.arange(9).reshape(3, 3), columns=list("abc"))
df1
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }
</style>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>a</th>
      <th>b</th>
      <th>c</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>1</td>
      <td>2</td>
    </tr>
    <tr>
      <th>1</th>
      <td>3</td>
      <td>4</td>
      <td>5</td>
    </tr>
    <tr>
      <th>2</th>
      <td>6</td>
      <td>7</td>
      <td>8</td>
    </tr>
  </tbody>
</table>
</div>




```python
df2 = DataFrame(np.arange(16).reshape(4, 4), columns=list("abcd"))
df2
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>a</th>
      <th>b</th>
      <th>c</th>
      <th>d</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>3</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4</td>
      <td>5</td>
      <td>6</td>
      <td>7</td>
    </tr>
    <tr>
      <th>2</th>
      <td>8</td>
      <td>9</td>
      <td>10</td>
      <td>11</td>
    </tr>
    <tr>
      <th>3</th>
      <td>12</td>
      <td>13</td>
      <td>14</td>
      <td>15</td>
    </tr>
  </tbody>
</table>
</div>




```python
df1 + df2
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>a</th>
      <th>b</th>
      <th>c</th>
      <th>d</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.0</td>
      <td>2.0</td>
      <td>4.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>7.0</td>
      <td>9.0</td>
      <td>11.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>14.0</td>
      <td>16.0</td>
      <td>18.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>

- 공통되지 않은 인덱스에는 역시 `NaN`을 반환하는 것을 볼 수 있다.



- `DataFrame` 역시 `.add()`메소드가 존재한다.


```python
df1.add(df2)
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>a</th>
      <th>b</th>
      <th>c</th>
      <th>d</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.0</td>
      <td>2.0</td>
      <td>4.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>7.0</td>
      <td>9.0</td>
      <td>11.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>14.0</td>
      <td>16.0</td>
      <td>18.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>



- `.add()`메소드의 적용 순서를 바꿔도 결과가 같음을 알 수 있다.


```python
df2.add(df1)
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>a</th>
      <th>b</th>
      <th>c</th>
      <th>d</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.0</td>
      <td>2.0</td>
      <td>4.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>7.0</td>
      <td>9.0</td>
      <td>11.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>14.0</td>
      <td>16.0</td>
      <td>18.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
df1.add(df2, fill_value=0)
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>a</th>
      <th>b</th>
      <th>c</th>
      <th>d</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.0</td>
      <td>2.0</td>
      <td>4.0</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>7.0</td>
      <td>9.0</td>
      <td>11.0</td>
      <td>7.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>14.0</td>
      <td>16.0</td>
      <td>18.0</td>
      <td>11.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>12.0</td>
      <td>13.0</td>
      <td>14.0</td>
      <td>15.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
df2.add(df1,fill_value=0)
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>a</th>
      <th>b</th>
      <th>c</th>
      <th>d</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.0</td>
      <td>2.0</td>
      <td>4.0</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>7.0</td>
      <td>9.0</td>
      <td>11.0</td>
      <td>7.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>14.0</td>
      <td>16.0</td>
      <td>18.0</td>
      <td>11.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>12.0</td>
      <td>13.0</td>
      <td>14.0</td>
      <td>15.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
df1.mul(df2, fill_value=1) # 곱 연산도 있다, 행렬 곱연산이 아님. 한칸씩 각각끼리의 value 곱
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>a</th>
      <th>b</th>
      <th>c</th>
      <th>d</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.0</td>
      <td>1.0</td>
      <td>4.0</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>12.0</td>
      <td>20.0</td>
      <td>30.0</td>
      <td>7.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>48.0</td>
      <td>63.0</td>
      <td>80.0</td>
      <td>11.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>12.0</td>
      <td>13.0</td>
      <td>14.0</td>
      <td>15.0</td>
    </tr>
  </tbody>
</table>
</div>



## 3. DataFrmae과 Series 간의 연산


```python
df = DataFrame(np.arange(16).reshape(4, 4), columns=list("abcd"))
df
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }
</style>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>a</th>
      <th>b</th>
      <th>c</th>
      <th>d</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>3</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4</td>
      <td>5</td>
      <td>6</td>
      <td>7</td>
    </tr>
    <tr>
      <th>2</th>
      <td>8</td>
      <td>9</td>
      <td>10</td>
      <td>11</td>
    </tr>
    <tr>
      <th>3</th>
      <td>12</td>
      <td>13</td>
      <td>14</td>
      <td>15</td>
    </tr>
  </tbody>
</table>
</div>




```python
s = Series(np.arange(10, 14), index=list("abcd"))
s
```


    a    10
    b    11
    c    12
    d    13
    dtype: int32




```python
df + s
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>a</th>
      <th>b</th>
      <th>c</th>
      <th>d</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>10</td>
      <td>12</td>
      <td>14</td>
      <td>16</td>
    </tr>
    <tr>
      <th>1</th>
      <td>14</td>
      <td>16</td>
      <td>18</td>
      <td>20</td>
    </tr>
    <tr>
      <th>2</th>
      <td>18</td>
      <td>20</td>
      <td>22</td>
      <td>24</td>
    </tr>
    <tr>
      <th>3</th>
      <td>22</td>
      <td>24</td>
      <td>26</td>
      <td>28</td>
    </tr>
  </tbody>
</table>
</div>

- `Series`의 `index` 값을 `DataFrame`의 `column_index`명에 맞추어 연산이 이루어졌다.
- 또한 `BroadCasting`으로 인하여 각 `column`의 모든 `rows`에 `Series`의 `Value`값이 각각 더해진 결과가 반환되었음을 알 수 있다.



> **그렇다면, 다음 예시처럼 `Series`의 `index`와 `DataFrame`의 `column_index` 값이 서로 아무것도 일치하지 않는다면 어떻게 될까?**




```python
s2 = Series(np.arange(10, 14))
s2
```


    0    10
    1    11
    2    12
    3    13
    dtype: int32




```python
df + s2
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>a</th>
      <th>b</th>
      <th>c</th>
      <th>d</th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>

- 서로 일치하지 않을 때, Series의 index가 DataFrame의 column_index로 추가되었음을 알 수 있다.
- `DataFrame`끼리의 연산에서 나타났던 현상을 볼 수 있다.
- 인덱스가 일치하지를 않아서 연산이 이루어지지 못한 부분에 대한 반환값은 역시 NaN이다.





> DataFrame과 Series 간의 연산에도 역시 add() 메소드가 있습니다.
>
> 다음 예시를 보겠습니다.

```python
print(df)
print(s2)
```

```
    a   b   c   d
0   0   1   2   3
1   4   5   6   7
2   8   9  10  11
3  12  13  14  15
0    10
1    11
2    12
3    13
dtype: int32
```



- `axis`를 인자로 입력받아 이를 기준으로 `row`인지 `column`인지 고려하며, 그에 해당하는 인덱스를 맞춰 메소드를 수행한다.
- 아래의 예시에서는 `axis=0`이므로 `Series`의 인덱스가 `row`를 나타낸다고 생각하여 메소드를 수행한다.


```python
df.add(s2, axis=0)
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }
</style>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>a</th>
      <th>b</th>
      <th>c</th>
      <th>d</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>10</td>
      <td>11</td>
      <td>12</td>
      <td>13</td>
    </tr>
    <tr>
      <th>1</th>
      <td>15</td>
      <td>16</td>
      <td>17</td>
      <td>18</td>
    </tr>
    <tr>
      <th>2</th>
      <td>20</td>
      <td>21</td>
      <td>22</td>
      <td>23</td>
    </tr>
    <tr>
      <th>3</th>
      <td>25</td>
      <td>26</td>
      <td>27</td>
      <td>28</td>
    </tr>
  </tbody>
</table>
</div>





- 또다른 예시로는 `axis=1`일 때, `Series`의 인덱스가 `column`을 나타낸다고 생각하여 메소드를 수행한다.


```python
df.add(s2, axis=1)
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

</style>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>a</th>
      <th>b</th>
      <th>c</th>
      <th>d</th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>

</div>

