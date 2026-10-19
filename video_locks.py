import os
import threading


_VIDEO_READ_LOCKS = {}
_VIDEO_READ_LOCKS_GUARD = threading.Lock()


def get_video_read_lock(video_path):
    abs_path = os.path.abspath(video_path)
    with _VIDEO_READ_LOCKS_GUARD:
        if abs_path not in _VIDEO_READ_LOCKS:
            _VIDEO_READ_LOCKS[abs_path] = threading.Lock()
        return _VIDEO_READ_LOCKS[abs_path]
