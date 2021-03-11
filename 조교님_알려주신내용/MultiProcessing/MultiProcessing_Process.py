# multiprocessing 모듈을 사용하면 스레딩 모듈로 스레드를 생성 할 수있는 것과 동일한 방식으로 프로세스를 생성 할 수 있습니다. 여기서 주요 포인트는 프로세스를 생성하기 때문에 GIL (Global Interpreter Lock)을 피하고 시스템의 여러 프로세서를 최대한 활용할 수 있다는 것입니다.



from multiprocessing import Process
import time
import os

# 시작시간
start_time = time.time()

# 멀티쓰레드 사용하기 (소수찾기)
def count(cnt):
    proc = os.getpid()
    for i in range(cnt):
        print(f"Process Id : {proc} -- {i}")

if __name__ == '__main__':
    # 멀티쓰레드 Process 사용하기
    num_arr = [100000, 100000, 100000, 100000]
    procs = []

    for number in num_arr:
        # Process 객체 생성
        proc = Process(target=count, args=(number,))
        procs.append(proc)
        proc.start()


    for proc in procs:
        proc.join()

# 사용시간
print(f"--- {time.time() - start_time} seconds ---")
