import contextlib
import signal
from datetime import datetime


def now_simpleformat() -> str:
    time = datetime.now().isoformat(timespec="seconds")
    time = time.replace("-", "")
    time = time.replace(":", "")
    time = time.replace("T", "-")
    return time


@contextlib.contextmanager
def defer_interrupt():
    signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGINT, signal.SIGTERM})
    try:
        yield
    finally:
        signal.pthread_sigmask(signal.SIG_UNBLOCK, {signal.SIGINT, signal.SIGTERM})


def byteswap(X):
    return X.view(X.dtype.newbyteorder()).byteswap()
