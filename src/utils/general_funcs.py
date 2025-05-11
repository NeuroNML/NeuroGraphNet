from datetime import datetime


def log(msg):
    """
    Log message with timestamp to monitor training.
    """
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)
