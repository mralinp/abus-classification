import time

def timer(main_function: callable):
    def inner(*args, **kwargs):
        start = time.time()
        res = main_function(*args, **kwargs)
        end = time.time()
        elapsed_time = end - start
        print(f"[{main_function.__name__}] execution time: {elapsed_time} seconds")
        return res
    return inner

@timer
def just_wait(duration:int, colleague_name:str):
    time.sleep(duration)
    print(f"ok {colleague_name}, let's go now")
    
if __name__ == '__main__':
    just_wait(duration=10, colleague_name='ali')