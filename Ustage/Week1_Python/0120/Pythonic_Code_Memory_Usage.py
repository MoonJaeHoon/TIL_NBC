
from memory_profiler import profile

d1 = {i:i for i in range(10000)}
d2 = {j:j for j in range(10001,20000)}

@profile(precision=4)
def for_loop(d1, d2):
    result = {}
    
    for k in d1:
        result[k] = d1[k]
    for k in d2:
        result[k] = d2[k]
        
    return result

@profile(precision=4)
def update_method(d1,d2):
    result = {}
    result.update(d1)
    result.update(d2)
    return result

@profile(precision=4)
def dict_comprehension(d1,d2):
    result = {k:v for d in [d1,d2] for k,v in d.items()}
    return result

@profile(precision=4)
def dict_kwargs(d1,d2):
    result = {**d1,**d2}
    return result

if __name__ == "__main__":
    data1 = for_loop(d1,d2)
    data2 = update_method(d1,d2)
    data3 = dict_comprehension(d1,d2)
    data4 = dict_kwargs(d1,d2)    