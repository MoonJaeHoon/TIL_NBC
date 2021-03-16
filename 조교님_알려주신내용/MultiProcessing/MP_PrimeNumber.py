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
    # input_range = range(1,10)  # 소수를 찾을 범위를 미리 선언해주었습니다.
    pool = Pool(4)  # num_thread = 4로 설정했습니다.
    
    # 선언해놓은 range에서 4개의 thread가 병렬로 수행합니다.
    # list comprehension & if문을 통해 None이 Return된 원소들은 필터링해주었습니다.
    answer_list = [p for p in pool.map(prime_number, input_range) if p!=None]
    print(answer_list)


# 사용시간
print(f"--- {time.time() - start_time} seconds ---")