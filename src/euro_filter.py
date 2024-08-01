import math
import numpy as np

class OneEuroFilter:
    def __init__(self, min_cutoff=1.0, beta=0.0, d_cutoff=1.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.x_prev = None
        self.dx_prev = None
        self.t_prev = None

    def alpha(self, cutoff, dt):
        tau = 1.0 / (2 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def smooth(self, x, alpha, x_prev):
        if x_prev is None:
            return x
        return alpha * x + (1 - alpha) * x_prev

    def __call__(self, x, t):
        if self.t_prev is None:
            self.t_prev = t
            self.x_prev = x
            self.dx_prev = 0.0
            return x

        dt = t - self.t_prev
        dx = (x - self.x_prev) / dt if dt > 0 else 0.0
        edx = self.smooth(dx, self.alpha(self.d_cutoff, dt), self.dx_prev)
        cutoff = self.min_cutoff + self.beta * abs(edx)
        alpha = self.alpha(cutoff, dt)
        filtered_x = self.smooth(x, alpha, self.x_prev)

        self.t_prev = t
        self.x_prev = filtered_x
        self.dx_prev = edx

        return filtered_x


if __name__ == "__main__":
    # Usage example:
    one_euro_filter = OneEuroFilter(min_cutoff=1.0, beta=0.0, d_cutoff=1.0)
    data = [1, 2, 3, 4, 5]  # Example data points
    filtered_data = [one_euro_filter(x, t) for t, x in enumerate(data)]
    print(filtered_data)
