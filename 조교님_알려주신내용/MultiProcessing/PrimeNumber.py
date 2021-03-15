
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
    input_number = 2*10**6
    answer_list = prime_number(input_number)
    print(answer_list)

# 사용시간
print(f"--- {time.time() - start_time} seconds ---")
