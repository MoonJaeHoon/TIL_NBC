
import time
import os

# 시작시간
start_time = time.time()

# input_number 주어졌을 때, 해당숫자 이하의 모든 소수찾기
# 결과는 리스트로 반환
def prime_number(input_number):
    prime_number_list = [] # 소수 저장 리스트
    for c in range(1,input_number+1):
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
    input_number = 2*10**6  # 2백만 이하의 모든 소수 구하기
    answer_list = prime_number(input_number)
    print(answer_list)

# 사용시간
print(f"--- {time.time() - start_time} seconds ---")
