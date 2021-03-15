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
    input_range = range(1,2*10**6)
    pool = Pool(4)
    pool.map(prime_number, input_range)

    # answer_list = prime_number(input_number)
    # print(answer_list)


# 사용시간
print(f"--- {time.time() - start_time} seconds ---")