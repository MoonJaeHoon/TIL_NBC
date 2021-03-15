## 일반 Processing

```python

import time
import os

# 시작시간
start_time = time.time()

# input_number 주어졌을 때, 소수찾기
def prime_number(input_number):
    prime_number_list = []
    for c in range(1,input_number+1):
        count=0
        for i in range(1,int(c**(0.5))+1):
            if c%i==0:
                count+=1
        if count==1:
            prime_number_list.append(c)
    return prime_number_list

if __name__ == '__main__':
    input_number = 10**4
    answer_list = prime_number(input_number)
    print(answer_list)

# 사용시간
print(f"--- {time.time() - start_time} seconds ---")

```

```
--- 156.34380269050598 seconds ---
```





## MultiProcessing

```py
from multiprocessing import Pool
import time
import os

# 시작시간
start_time = time.time()


def prime_number(input_number):
    count=0
    for i in range(1,int(input_number**(0.5))+1):
        if input_number%i==0:
            count+=1
    if count==1:
        return input_number
    else:
        return

if __name__ == '__main__':
    input_range = range(1,10**4)
    pool = Pool(4)
    pool.map(prime_number, input_range)

    # answer_list = prime_number(input_number)
    # print(answer_list)
# 사용시간
print(f"--- {time.time() - start_time} seconds ---")
```

```
--- 81.36539554595947 seconds ---
```



MultiProcessing으로 수행한 결과가 시간 측면에서 효율적이라는 것을 알 수 있었다.