# 연산자 우선순위
- 이번엔 아주 간단하지만 헷갈릴 수 있는 연산자의 우선순위에 대해 알아보겠습니다.
- 네이버부스트캠프 중 마스터님께서 올려주신 `Further Question`이었습니다.

```py
print(4!=0 not in [1,2,3])
print((4!=0) not in [1,2,3])
print(1!=0 not in [1,2,3])
print(4!=0 not in [0,1,2,3])
print((4!=0) not in [0,1,2,3])
```

```
True
False
True
False
False
```

#
-  Note that `a op1 b op2 c` doesn’t imply any kind of comparison between a and c, so that,
-  e.g.,` x < y > z` is perfectly legal (though perhaps not pretty).

`출처 : https://www.notion.so/A-B-C-A-B-and-B-C-1a10d1864e0c4be8b3fb2ac9d173f260 `

#
- 위 설명을 간단하게 이해하자면 "`피연산자a` **`연산자1`** `피연산자b` **`연산자2`** `피연산자c`를 푸는 문제에서는 이를 만족시키는 조건으로 세분화해서 푸는 것이 합리적이라는 것이다.
- 따라서 `피연산자a` **`연산자1`** `피연산자b` 와 `피연산자b` **`연산자2`** `피연산자c`를 모두 고려하여 조건을 생각할 수 있어야 한다는 것이다.
#
> **결국 위 문제에서의 풀이코드는 다음과 같이 적을 수 있겠다**
```py
print( (4 != 0) and (0 not in [1, 2, 3]) ) #1
print( (4 != 0)  not in [1, 2, 3] ) #2
print( (1 != 0) and (0 not in [1, 2, 3]) ) #3
print( (4 != 0) and (0 not in [0,1, 2, 3]) ) #4
print( (4 != 0) not in [0,1, 2, 3] ) #5
```

```
True
False
True
False
False
```

- 당연하게도, 괄호가 쳐져있는 부분은 괄호 안의 해당 연산을 먼저 수행한 값이 그 다음 연산의 피연산자가 되겠다.