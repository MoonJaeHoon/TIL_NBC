# ListComprehension vs Map

- 코딩을 하면서 ListComprehension 과 Map 뭐가 더 효율적인걸까? 이런 궁금증이 생겼다.
- 결과를 말하자면 메모리적인 부분, 시간적인 부분 모두에서 map이 더 효율적이다.
- 요약 : map이 generator이고 list는 iterator이기 때문.



## 1번 : hex func에 대한 map vs ListComprehension
- `input`은 10개, 시간측정을 위해서는 `timeit`모듈의 `timeit` 함수를 사용하였다.

```python
import timeit
xs=range(10)
print(timeit.timeit(stmt=f'map(hex, {xs})', number=10_000_000))
print(timeit.timeit(stmt=f'[hex(x) for x in {xs}]', number=10_000_000))
```

```
python ListComprehension_vs_Map.py
3.1796699999999998
13.150395399999997
```
- map이 약 4배 더 빠르다.



## 2번: 단순 +연산에 대한 map vs ListComprehension
- `input`은 0부터 99까지로 100개이다(1번보다 데이터 수를 늘렸다.)
```python
import timeit
xs=range(100)
print(timeit.timeit(stmt=f'map(lambda x: x+10, {xs})', number=10_000_000))
print(timeit.timeit(stmt=f'[x+10 for x in {xs}]', number=10_000_000))
```

```
python ListComprehension_vs_Map.py
3.6490641000000004
52.1029305
```
- map이 약 15배 더 빠르다.

> **왜 이런 결과가 나올까?**
- 앞서 설명했듯, map은 generator를 생성하기 때문이다.
- generator type은 메모리에 모든 값을 저장해놓는 것이 아니라 주어져있는 크기만을 기억하고, 현재 값만을 저장하고 있는 객체이다
- 만약, 현재 값을 반환하게 되면 주소로만 기억하고 있던 다음 값을 저장하고 있는 상태가 되며, 이 과정을 반복한다.
- 따라서 시간, 메모리 면에서 매우 이점을 가질 수 있는 것이다.




## 3번: Convert to List after mapping vs ListComprehension
- 그렇다면, 만약 map 결과에 list변환을 해주는 연산까지 더한다면 어떤 결과가 나올 것 같은가?
- 당연히 generator를 생성해주는 map을 활용해준 의미가 없이 iterator로 변환하는 과정이 추가되게 되므로 이점이 없고, 오히려 연산과정이 늘어 손해를 보게 되는 것을 볼 수 있다.

```python
import timeit
xs=range(100)
print(timeit.timeit(stmt=f'list(map(lambda x: x+10, {xs}))', number=10_000_000))
print(timeit.timeit(stmt=f'[x+10 for x in {xs}]', number=10_000_000))
```

```
python ListComprehension_vs_Map.py
104.1057988
52.91681160000002
```