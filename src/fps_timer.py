import time

class FPS:
    def __init__(self):
        self.start_time = None
        self.total_num_frames = 0
        self.num_frames = 0
        self.fps = 0

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    def start(self):
        if self.start_time is not None:
            raise RuntimeError("FPS timer is already running")
        self.start_time = time.time()
        return self

    def update(self):
        self.total_num_frames += 1
        self.calc_fps()
        
    def get_fps(self):
        if self.start_time is None:
            raise RuntimeError("FPS timer is not running")
        end_time = time.time()
        elapsed_time = end_time - self.start_time
        fps = self.num_frames / elapsed_time
        return round(fps, 2)

    def stop(self):
        fps = self.get_fps()
        self.reset()
        return fps

    def reset(self):
        self.start_time = None
        self.num_frames = 0

    def calc_fps(self):
        self.num_frames += 1
        if self.start_time is None:
            self.start_time = time.time()
        else:
            elapsed_time = time.time() - self.start_time
            if elapsed_time > 1:
                self.fps = self.num_frames / elapsed_time
                self.start_time = time.time()
                self.num_frames = 0

    def __str__(self):
        return f"FPS: {self.get_fps()}"