from time import time
from contextlib import contextmanager

@contextmanager
def timed(msg: str):
  print(msg)
  start = time()
  yield
  stop = time()
  print(f'({stop - start:.1f}s)')