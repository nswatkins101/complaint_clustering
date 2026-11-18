import functools
from joblib import memory

cachedir = mkdtemp()
memory = Memory(location=cachedir, verbose=10)

def long_func():
    print("hello world")
    if x>1:
       return x*long_func(x-1)*long_func(x-2)
    else:
        return 1
        

start_time = timer()

long_func_cache = functools.cache(long_func)
long_func_joblib_cache = memory.cache(long_func)
long_func(100)

end_time = timer()

print(f'time to run was {end_time-start_time}')