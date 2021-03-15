

## 일반 Processing

```python

import time
import os

# 시작시간
start_time = time.time()

# input_range 주어졌을 때, 해당 범위에서 모든 소수찾기
# 결과는 리스트로 반환
def prime_number(input_range):
    prime_number_list = [] # 소수 저장 리스트
    for c in input_range:
        count=0 # 약수의 개수 카운팅
        # 약수를 찾을 때에는 본인의제곱근까지만 찾으면 된다.
        for i in range(1,int(c**(0.5))+1):  
            if c%i==0:
                count+=1    # 약수라면 카운팅+1
        # 카운팅이 1이라면 소수로 판별
        if count==1:
            prime_number_list.append(c)
    return prime_number_list

if __name__ == '__main__':
    input_range = range(2,2*10**6+1)  # 2백만 이하의 모든 소수 구하기
    answer_list = prime_number(input_range)
    print(answer_list)

# 사용시간
print(f"--- {time.time() - start_time} seconds ---")

```

```
--- 156.34380269050598 seconds ---
```





## MultiProcessing

```python
from multiprocessing import Pool
import time
import os

# 시작시간
start_time = time.time()

def prime_number(input_number): # input_number가 들어왔을 때 소수라면 해당 숫자를 출력하는 함수
    count=0
    for i in range(1,int(input_number**(0.5))+1):   # 약수가 몇개인지 카운팅함.
        if input_number%i==0:
            count+=1

    if count==1: # count가 1이면 소수로 판별
        return input_number # 소수라면, 해당 값을 return

if __name__ == '__main__':
    input_range = range(2,2*10**6)  # 소수를 찾을 범위를 미리 선언해주었습니다.
    pool = Pool(4)  # num_thread = 4로 설정했습니다.
    answer_list = pool.map(prime_number, input_range) # 선언해놓은 range에서 4개의 thread가 병렬로 수행합니다.
    print(answer_list)


# 사용시간
print(f"--- {time.time() - start_time} seconds ---")
```

```
--- 0.0 seconds ---
--- 0.0 seconds ---
--- 0.0 seconds ---
--- 0.0 seconds ---
--- 70.1404619216919 seconds ---
```



MultiProcessing으로 수행한 결과가 시간 측면에서 효율적이라는 것을 알 수 있었다.



**※ 수정필요.. MultiProcessing에서 if 조건문을 조금 다르게 걸어야 할 것 같아보인다.**