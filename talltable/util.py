import contextlib
import signal


@contextlib.contextmanager
def defer_interrupt():
    signal.pthread_mask(signal.SIG_BLOCK, {signal.SIGINT, signal.SIGTERM})
    try:
        yield
    finally:
        signal.pthread_mask(signal.SIG_UNBLOCK, {signal.SIGINT, signal.SIGTERM})

