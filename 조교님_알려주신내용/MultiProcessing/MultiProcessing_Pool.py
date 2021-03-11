import time

start_time = time.time()

from multiprocessing import Pool

def f(x):
    return x*x

num_list = list(range(10**6))

if __name__ == '__main__':
    with Pool(5) as p:
        p.map(f, num_list)

print(f"--- {time.time() - start_time} ---")
