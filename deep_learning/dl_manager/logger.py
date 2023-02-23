import contextlib
import logging
import sys
import time

ROOT = 'DL-Pipeline'

formatter = logging.Formatter('[{name}][{asctime}][{levelname}]: {msg}',
                              style='{')
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(formatter)

global_log = logging.getLogger(ROOT)
global_log.addHandler(handler)
global_log.setLevel(logging.DEBUG)


def get_logger(name: str) -> logging.Logger:
    full_name = f'{ROOT}.{name}'
    return logging.getLogger(full_name)



@contextlib.contextmanager
def timer(name):
    log = get_logger(f'{name}.Timer')
    log.info(f'Measuring runtime of {name}')
    start = time.time()
    yield
    log.info(f'Finished Measuring runtime of {name}. Took {time.time() - start} seconds.')

