from contextlib import contextmanager
from timeit import default_timer as timer
import json


@contextmanager
def snapshot(file_name, ns):
    with open(file_name, "a+") as f:
        # It goes to the last line
        f.seek(0)
        try:
            data = json.load(f)
        except Exception:
            data = {}

    if ns not in data:
        data[ns] = {}

    start = timer()

    yield data[ns]

    end = timer()
    data[ns]["elapsed_time"] = end - start
    with open(file_name, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    return data
