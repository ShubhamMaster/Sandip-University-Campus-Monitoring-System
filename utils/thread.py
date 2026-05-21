import cv2
import threading


class ThreadingClass:
    """Threaded video capture — always returns the latest frame with no buffering lag."""

    def __init__(self, name):
        self.cap = None
        self._stopped = False
        if isinstance(name, int):
            for backend in (cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY):
                cap = cv2.VideoCapture(name, backend)
                if cap is not None and cap.isOpened():
                    ret, _ = cap.read()
                    if ret:
                        self.cap = cap
                        break
                    cap.release()
            if self.cap is None:
                self.cap = cv2.VideoCapture(name)
        else:
            self.cap = cv2.VideoCapture(name)

        # Reduce OpenCV internal buffer to 1 frame
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self._frame = None
        self._lock = threading.Lock()
        t = threading.Thread(target=self._reader, daemon=True)
        t.start()

    def _reader(self):
        while not self._stopped:
            ret, frame = self.cap.read()
            if not ret:
                self._stopped = True
                break
            with self._lock:
                self._frame = frame

    def read(self):
        """Return the most recent frame (never blocks on stale buffer)."""
        with self._lock:
            return self._frame

    def release(self):
        self._stopped = True
        self.cap.release()
