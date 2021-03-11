import time

start_time = time.time()

def f(x):
    return x*x

num_list = list(range(10**6))
if __name__ == '__main__':
    [f(x) for x in num_list]

print(f"--- {time.time() - start_time} ---")
