import numpy as np
from numpy import arange, meshgrid, exp, cos, pi, sqrt, e



class Source():
    def __init__(self, sample_size, resolution, randomize):
        self.sample_size = sample_size
        self.resolution = resolution
        self.randomize = randomize
        self.signal = np.random.randint(0, 1) if self.randomize else 1

    def generate_arena(self):
        """To be run at every reset. This will generate data for the sampler."""
        self.lb = [-5, -5] if not self.randomize else np.random.uniform(0, 1, size=2) * -25
        self.ub = [5, 5] if not self.randomize else np.random.uniform(0, 1, size=2) * 25
        self.data = self.get_data(self.lb, self.ub, self.sample_size, self.signal)
    
    def get_data(self, lb = [-5, -5], ub = [5, 5], resolution=100, signal=1):
        """
        input:
        -lb: Lower bound for the env.
        -ub: Upper bound for the env.
        -resolution: Shape of the functional space (resolution, resolution) 
        -signal: Decides which source environment to use. 1 for the unimodal and 0 for the multimodal.

        output:
        -data: Array of shape (resolution**2, 3) containing locations X1, X2 and the signal measurements Y
        """
        x1 = np.linspace(lb[0], ub[0], resolution)
        x2 = np.linspace(lb[1], ub[1], resolution)
        X1, X2 = np.meshgrid(x1, x2)
        Y, self.a, self.b = self.unimodal_signal(X1, X2) if signal == 1 else self.multimodal_signal(X1, X2)
        data = np.hstack((X1.reshape(-1, 1), X2.reshape(-1, 1), Y.reshape(-1, 1)))
        return data

    def unimodal_signal(self, x1, x2, measure=False):
        """Returns unimodal signal output on given x1 and x2 values."""
        if measure:
            return (x1+self.a)**2 + (x2+self.b)**2

        a = np.random.uniform(-2.50, 2.50)
        b = np.random.uniform(-3.00, 3.00)
        return (x1+a)**2 + (x2+b)**2, a, b

    def multimodal_signal(self, x1, x2, measure=False):
        """Returns multimodal signal output on given x1 and x2 values."""
        if measure:
            return -20.0 * exp(-0.2 * sqrt(0.5 * ((x1+self.a)**2 + (x2+self.b)**2))) - exp(0.5 * (cos(2 * pi * (x1+self.a)) + cos(2 * pi * (x2+self.b)))) + e + 20

        a = np.random.uniform(-3.00,3.00)
        b = np.random.uniform(-3.00,3.00)
        return -20.0 * exp(-0.2 * sqrt(0.5 * ((x1+a)**2 + (x2+b)**2))) - exp(0.5 * (cos(2 * pi * (x1+a)) + cos(2 * pi * (x2+b)))) + e + 20, a, b

    def measure_signal(self, locs):
        """Returns measured signal for given x1 and x2 values."""
        X1, X2 = locs[:, 0], locs[:, 1]
        signal =  self.unimodal_signal(X1, X2, measure=True) if self.signal == 1 \
        else self.multimodal_signal(X1, X2, measure=True)

        return np.hstack((X1.reshape(-1, 1), X2.reshape(-1, 1), signal.reshape(-1, 1)))

    def get_info(self):
        return self.data, self.lb, self.ub