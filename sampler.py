import numpy as np


class Sampler():
    def __init__(self, sample_cap, velocity=0.1):
        self.sample_cap = sample_cap
        self.velocity_mps = velocity

    def reset(self, source):
        """
        input:
        -source: Source environment object which contains functional space information (X1 X2 and measured signal, as well as upper and lower bounds).

        output:
        -D_old: A random sampleset of size (self.sample_cap, 3) to start with.
        -D_new: A random amount of new observations collected close to that of D_old.
        """

        data, self.lb, self.ub = source.get_info()
        random_idxs = np.random.permutation(data.shape[0])

        D_old = data[random_idxs[:self.sample_cap]]

        D_new = self.create_new_obs(D_old)

        return D_old, D_new

    def create_new_obs(self, D_old):
        """
        steps:
        1. Generate D_new size which will be a random length in [0.1*self.sample_sample, 0.5*self.sample_size). 
            -This will simulate robots gathering different samples per step.
        2. A set of points are chosen from D_old using D_new's size. These points will be used to create D_new.
        3. Random headings are created in all directions using random values.
        4. The movement time is a random number between 0 and 1 second.
        5. Total distance traveled for the step is calculated as the robot velocity (const) multiplied by the movement time (random).
        6. D_new is created using P* = P + H*D where:
            -P* and P are the new and old points respectively.
            -H is the robot heading.
            -D is the distance traveled.
            -The result is clipped between the upper and lower bounds.
        
        input:
        -D_old: Either a random permutation of the functional space (reset) or a combination of {D_old_filt, D_new_filt} (step).

        output:
        -D_new: New observations for the CNN to analyze.
        
       """

        # Amount of new observations to take will be between the sample and action cap
        obs_sample_size = int(np.random.uniform(0.1, 0.5) * self.sample_cap)
        
        np.random.shuffle(D_old)

        old_points = D_old[:obs_sample_size]

        # Generate random headings for the *robots* to travel
        random_headings = np.random.uniform(-1.0, 1.0, size=(obs_sample_size, 2))
        movement_time = np.random.uniform()
        dist_traveled = movement_time * self.velocity_mps

        new_obs = old_points[:, :2] + random_headings * dist_traveled

        return np.clip(new_obs, self.lb, self.ub)
