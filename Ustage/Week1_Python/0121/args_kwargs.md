# *args & **kwargs
# 1. *args 함수
- 여기서, `args`란 **arguments**이며 말 그대로 변수들을 여러 개 받는다는 의미 입니다.
- 그러니 변수들을 여러개(몇 개인지 몰라도 문제가 없는) 받아도 이를 입력으로 받아 처리할 수 있는 함수를 정의할 수 있게 됩니다.
- 다음 예시에서 해당 개념을 활용해 여러 인자들을 입력으로 받는 함수를 구현해보았습니다.

```py
def index_dict(*args):
    my_dict = {}
    for ind,val in enumerate(args):
        my_dict[ind] = val
    return my_dict

print(index_dict('어머니','아버지','누나','형','나'))
```
```
{0: '어머니', 1: '아버지', 2: '누나', 3: '형', 4: '나'}
```

## 예시 : 1.1 product 함수

- `*args` 인자로 여러 원소들을 `input`으로 받고, 받은 각각의 원소들에서 하나씩 뽑아 가능한 모든 조합 경우의 수(**데카르트 곱**이라고도 합니다)를 만들어 반환해줍니다.

```py
def product(*args, repeat=1):
    # product('ABCD', 'xy') --> Ax Ay Bx By Cx Cy Dx Dy
    # product(range(2), repeat=3) --> 000 001 010 011 100 101 110 111
    pools = [tuple(pool) for pool in args] * repeat
    result = [[]]
    for pool in pools:
        result = [x+[y] for x in result for y in pool]
    for prod in result:
        yield tuple(prod)
```
`출처 : https://docs.python.org/ko/3/library/itertools.html#itertools.product`
#
#
- 다음 예시에서는

$$
 _3C_1 *  _4C_1 * _3C_1 =36
$$

​	가지의 조합을 만들어 주는 것을 볼 수 있습니다.

```python
from itertools import product


items = [['a', 'b', 'c,'], ['1', '2', '3', '4'], ['!', '@', '#']]

product(*items)
```

```
<itertools.product at 0x138364e5480>
```

- 함수를 적용하면 반환값이 우리에게 익숙치 않은 `generator` 형태입니다.
- 이는 리스트로 변환해주면 우리가 알고있는 자료구조인 리스트로 보여집니다.



```python
from itertools import product

items = [['a', 'b', 'c,'], ['1', '2', '3', '4'], ['!', '@', '#']]
list(product(*items))
```

```
[('a', '1', '!'), ('a', '1', '@'), ('a', '1', '#'), ('a', '2', '!'), ('a', '2', '@'), ('a', '2', '#'), ('a', '3', '!'), ('a', '3', '@'), ('a', '3', '#'), ('a', '4', '!'), ('a', '4', '@'), ('a', '4', '#'), ('b', '1', '!'), ('b', '1', '@'), ('b', '1', '#'), ('b', '2', '!'), ('b', '2', '@'), ('b', '2', '#'), ('b', '3', '!'), ('b', '3', '@'), ('b', '3', '#'), ('b', '4', '!'), ('b', '4', '@'), ('b', '4', '#'), ('c,', '1', '!'), ('c,', '1', '@'), ('c,', '1', '#'), ('c,', '2', '!'), ('c,', '2', '@'), ('c,', '2', '#'), ('c,', '3', '!'), ('c,', '3', '@'), ('c,', '3', '#'), ('c,', '4', '!'), ('c,', '4', '@'), ('c,', '4', '#')]
```


## 예시 : 1.2 chain 함수
- 여러 개의 `iterable` 객체를 `input`으로 받습니다.
- 첫 번째 `iterable`에서 소진될 때까지 요소를 반환한 다음 이터러블로 넘어가고, 이런 식으로 `iterables`의 모든 이터러블이 소진될 때까지 진행하는 이터레이터를 만듭니다.
- 여러 시퀀스를 단일 시퀀스처럼 처리하는 데 사용됩니다

```py
def chain(*iterables):
    # chain('ABC', 'DEF') --> A B C D E F
    for it in iterables:
        for element in it:
            yield element
```
`출처 : https://docs.python.org/ko/3/library/itertools.html#itertools.chain`

#
- 문자열 또한 `iterable` 객체이므로 다음과 같이 input으로 받을 수 있다.
- 역시 반환값은 `generator`이기 때문에 `list`로 변환해주었다.
```py
from itertools import chain
print(chain('ABC', 'DEF'))
print(list(chain('ABC', 'DEF')))
```
```
<itertools.chain object at 0x0000022641A92130>
['A', 'B', 'C', 'D', 'E', 'F']
```


# 2. *kwargs
- 그럼 `*kwargs`는 무슨 뜻일까? 바로 **keyword arguments**의 줄임말이다.
- 말 그대로 `arguments`를 여러 개 받는데 `key`를 같이 입력값으로 받아서 처리하겠다는 뜻입니다.
- `python`에서 이처럼 `key`와 `value`를 활용하는 자료구조는 바로 `dict`이기 때문에 `input` 값을 `dict type`로 간주하여 처리하는 함수인 것입니다.

``` py
def how_old(**kwargs):
    for k,v in kwargs.items():
        if v<=27:
            print(f'{k}는 {v}살입니다. 어리네요')
        else:
            print(f'{k}는 {v}살입니다. 먹을만큼 먹었네요')

print(how_old(Jerry=30,Tom=35,SpongeBob=21,ZZANG9=7))
```

```
Jerry는 30살입니다. 먹을만큼 먹었네요
Tom는 35살입니다. 먹을만큼 먹었네요
SpongeBob는 21살입니다. 어리네요
ZZANG9는 7살입니다. 어리네요
```

## 예시 2.1 : dict.update() 메소드
- `dict type` 데이터를 다루는데 있어서 매우 유용한 메서드이다.
- for문과 같은 반복문을 적용할 필요없이, `update()` 메서드를 이용하면 원래 없었던 `key value`는 추가해주는 것은 물론이고 이전에 가지고 있던 `key`와 겹치면 해당 `value`를 수정하여 업데이트해준다.
- 대용량 `dict`를 업데이트해줄 때 매우 유용하며, 연산 복잡도 면에서까지도 `dict comprehension`보다 `Pythonic`하다는 것이 저명하다.
```py
def index_dict(*args): ## 메서드 적용을 위한 dict 생성 코드
                       ## kwargs랑 상관없음
    my_dict = {}
    for ind,val in enumerate(args):
        my_dict[ind] = val
    return my_dict

my_dict1 = index_dict("A","B","C","D","E","F","G","H","I",)
print(my_dict1)

my_dict2 = index_dict("AAA","BBB","CCC")
print(my_dict2)
```

```
{0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I'}
{0: 'AAA', 1: 'BBB', 2: 'CCC'}
```
- 우선, `dict.update()` 메서드는 인스턴스 객체(`dict1`), 메서드의 인자 (`dict2`)가 필요하기 때문에 위에서 `dict type` 자료를 2개 생성한 것이다.


```py
my_dict1.update(my_dict2)
print(my_dict1)
```
```
{0: 'AAA', 1: 'BBB', 2: 'CCC', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I'}
```
- 결과를 보면, 0 1 2 의 `key`에 해당하는 `value`는 수정되어 업데이트되었다.