import multiprocessing
import os
import time

def update_moving_average(mean, new_val, n):
    if n == None:
        return new_val
    mean *=  n
    mean += new_val
    mean /= n + 1
    return mean

def find_project_dir():
    curr_path, _ = os.path.split(os.path.realpath(__file__))
    return os.path.join(curr_path, "..")

def timit(func):
    """
    Times a function and returns a tuple, the result of the
    function and the time in seconds it took to run. Usage:

    @timeit
    def mylongfunc():
          pass

    From https://gist.github.com/bbengfort/a91680c79a62bcc93ea1
    """
    def wrapper(*args, **kwargs):
          start  = time.time()
          result = func(*args, **kwargs)
          finit  = time.time()
          print(finit-start)
          return result
    return wrapper

def timeit(f, s):
    """
    Helper function to time a function and print a string before and
    after execution.

    Input:
        f - Function with 0 arguments. The output of this function wil
            be returned from timeit.
        s - String, printed before running f() as "S..." and after
            running f() as "Done s..."

    Output:
        x - output of f()
    """
    big_s = s[0].upper() + s[1:]
    small_s = s[0].lower() + s[1:]

    print("{}...".format(big_s))

    t = time.time()
    x = f()

    print("Done {} in {}s.".format(small_s, time.time() - t))

    return x

def parmap(f, X, nprocs=multiprocessing.cpu_count()):
    """
    Paralellized map. 

    Input:
        f - monoid to be applied to each element of X
        X - iterable with each element in the domain of f

    Output:
        y - equivalent to [f(x) for x in X]

    Note: Code adapted from 
    http://stackoverflow.com/revisions/16071616/9.
    """

    def worker(f, q_in, q_out):
        while True:
            i, x = q_in.get()
            if i is None:
                break
            q_out.put((i, f(x)))


    q_in = multiprocessing.Queue(1)
    q_out = multiprocessing.Queue()

    worker_args = (f, q_in, q_out)
    new_worker = lambda: multiprocessing.Process(target=worker, 
                                                 args=worker_args)

    proc = [new_worker() for _ in range(nprocs)]
    for p in proc:
        p.daemon = True
        p.start()

    sent = [q_in.put((i, x)) for i, x in enumerate(X)]
    [q_in.put((None, None)) for _ in range(nprocs)]
    res = [q_out.get() for _ in range(len(sent))]

    [p.join() for p in proc]

    return [x for i, x in sorted(res)]