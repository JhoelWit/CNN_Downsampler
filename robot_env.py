from sampler import Sampler
from robot_source import Source
from custom_gp import CustomGP
import time

import gym
from gym import spaces
import numpy as np
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
from sklearn.feature_selection import mutual_info_regression
from sklearn.decomposition import PCA
from scipy.stats import entropy

class RobotEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    def __init__(self):
        super(RobotEnv, self).__init__()
        cont_bound = np.finfo(np.float32).max
        self.sample_cap = 100
        self.resolution = 100
        self.termination_iter = 1000
        termination_threshold = 0.10
        action_cap = 0.50
        self.termination_states = int(self.termination_iter * termination_threshold)
        self.action_cap = self.sample_cap + int(action_cap * self.sample_cap)

        self.action_space = spaces.Box(low=-cont_bound, high=cont_bound, 
                                        shape=(self.action_cap,), dtype=np.float32)
        self.observation_space = spaces.Dict(
                                            dict(
                                                matrix = spaces.Box(low=-cont_bound, high=cont_bound,
                                                shape=(3, self.resolution, self.resolution), dtype=np.float32),
                                                features = spaces.Box(low=-cont_bound, high=cont_bound,
                                                shape=(self.sample_cap, 3), dtype=np.float32)))
                                                # mask = spaces.Box(low=0.0, high=1.0, 
                                                # shape=(self.action_cap,), dtype=np.float32)))
          
        # GP Options
        self.kernel = ConstantKernel() * RBF(length_scale=1.0, length_scale_bounds=(1e-05, 50*10))
        self.kwargs = dict(alpha=1e-10, copy_X_train=True,\
                    kernel=self.kernel, n_restarts_optimizer=0, optimizer='fmin_l_bfgs_b')

        # Sampler for trajectory optimization and path generation
        self.randomize = True
        self.source = Source(self.sample_cap, self.resolution, self.randomize)
        self.sampler = Sampler(self.sample_cap, velocity=0.1)

        #PCA for mutual information reward
        self.pca = PCA(n_components=1)
        self.mutual_info_threshold = 0.80

    def step(self, action):
        """
        steps:
        1. The action is received and masked.
        2. If any new obs were filtered, then the new obs are all measured so the signals can be used later for fitting the GP
        3. After {D_old_filt, D_new_filt} is created, D_new is chosen from the sampler. Then, a new observation is created.
        4. The Y_mean and Y_std from the obs is used for calculating the reward.

        inputs:
        -action: Action provided by the CNN policy, consists of N "actions" after masking, which are chosen indices for the robot observation. N <= action cap.
        
        outputs:
        -observation: A dict consisting of three matrices: Y_mean, Y_std and new_obs Generated by the sampler. A mask is also included.
        -reward: Reward calculated by the Y_mean and/or Y_std matrices.
        -done: Returns True if a certain number of steps has passed AND the agent stays below a poor state threshold, else False.
        -info: Nothing, as of now.
        """
        
        self.iter += 1 
        
        step0 = time.time()
        while len(action) == 1:
            action = action[0]
        action = action[:self.D_old.shape[0] + self.D_new.shape[0]] # Used in place of masking
        max_idxs = np.argpartition(action, -self.sample_cap)[-self.sample_cap:]

        self.temp = self.D_old # Keeping the old dataset to calculate mutual information

        if any(x > self.sample_cap - 1 for x in max_idxs): 
            D_new_full = self.source.measure_signal(self.D_new)
            D_all = np.vstack((self.D_old, D_new_full))
            self.D_old = D_all[max_idxs]
        time_to_reorder = time.time() - step0

        self.D_new = self.sampler.create_new_obs(self.D_old)
        obs, mixed_state = self.create_obs()

        # reward = self.calculate_reward_entropy(mixed_state)
        reward = self.calculate_reward_mi()

        done = False if (self.iter < self.termination_iter) else True

        info = {"reorder_time":time_to_reorder,
                "poor_states":self.poor_states}
        # print("step time", time.time() - step0)
        return obs, reward, done, info

    def calculate_reward_entropy(self, mixed_state):
        """        
        input:
        -mixed_state: Array storing the Y_mean and Y_std arrays predicted from the gp.

        output:
        -reward: Normalized sum of non-positive terms in the change of entropy.

        additional:
        if the normalized sum is less than 10% of the entire matrix, the reward is instead -1. The hope is the agent learns to avoid under generalizing. 
        """
        reward = 0
        self.state_entropy = self.calculate_entropy(mixed_state)
        reward = np.sum((self.state_entropy - self.last_state_entropy) < 0) / self.resolution**2

        if reward <= 0.1:
            self.poor_states += 1
            reward = -1
        
        self.last_state_entropy = self.state_entropy

        return reward
    
    def calculate_reward_mi(self):
        """Attempts to calculate mutual information of the current dataset, in order to see if there has been information gain.
        Equation used is from here: https://journals.aps.org/pre/pdf/10.1103/PhysRevE.69.066138 equation 11.
        """

        mutual_info = self.calculate_mi()
        
        # print("mutual information", mutual_info)

        # if mutual_info < self.last_state_mi:
        #     reward = -1
        #     self.poor_states += 1
        # else:
        #     reward = 1

        # reward = mutual_info - self.last_state_mi

        # We'll use a threshold instead, the agent doesn't need to maximize this.
        # if mutual_info > self.mutual_info_threshold:
        #     reward = 1
        # else:
        #     reward = 0
        #     self.poor_states += 1

        reward = -abs(mutual_info - self.mutual_info_threshold)
        if reward < -1e-3:
            self.poor_states += 1

        # if reward < 0:
        #     self.poor_states += 1

        self.last_state_mi = mutual_info

        return reward

    def reset(self):
        """
        steps:
        1. Data and benchmark are generated
        2. D_old is generated from the sampler, as well as D_new
        3. The observation data is created
        """
        self.iter = 0
        self.poor_states = 0

        self.gp = CustomGP(max_iter=5e05, gtol=1e-06, resolution=self.resolution, **self.kwargs)

        self.source.generate_arena() 
        self.D_old, self.D_new = self.sampler.reset(self.source) 

        obs, self.last_state = self.create_obs()
        self.last_state_entropy = self.calculate_entropy(self.last_state)

        self.last_state_mi = self.calculate_mi()

        return obs 

    def create_obs(self):
        """The Observation consist of four things:
        1. The binary image of the new observations
        2. The Y_mean matrix
        3. The Y_std matrix
        4. The mask
        """
        obs = {}

        data, lb, ub = self.source.get_info()
        mixed_state = [*self.gp._fit_gp(self.D_old, data[:,:2])]

        binary_img = self.create_binary(lb, ub)

        obs["matrix"] = np.array((binary_img, mixed_state[0], mixed_state[1]))
        obs["features"] = self.D_old
        # print("mean ", mixed_state[0],"\n standard deviation",mixed_state[1])

        # mask = np.zeros((self.action_cap))
        # mask[-self.D_new.shape[0]:] = 1
        # obs["mask"] = mask

        # print(obs["mask"])

        return obs, mixed_state

    def create_binary(self, lb, ub):
        """
        Steps:
        1. blank_image template is created with zeros. The ranges are discretized into step sizes using the resolution. i.e -5 to 5 is discretized into 1000 steps.
        2. For each observation in D_new:
            -The x1 and x2 coordinates are plotted as i and j in the blank_image using the step size. 
                -Two cases are considered where a coordinate is less than and greater than zero.

        input:
        -lb: Lower bound for the env
        -ub: Upper bound for the env

        output:
        -blank_image: zero matrix of shape (self.resolution, self.resolution) where the new observations are plotted as 1's. 
        """
        obs, N = self.D_new, self.resolution

        if np.size(obs) == 0:
            return np.zeros((N, N))

        blank_image = np.zeros((N, N))
        step_size_x1 = (abs(lb[0]) + abs(ub[0])) / N
        step_size_x2 = (abs(lb[1]) + abs(ub[1])) / N

        # The following command will map the new observations to indices of a zero matrix representing the arena using a discretized step size.
        ij_obs = np.array([[int(abs(lb[0]) / step_size_x1) + int(abs(coord[0]) / step_size_x1) - 1 if coord[0] > 0 else int(abs(lb[0]) / step_size_x1) - int(abs(coord[0]) / step_size_x1),
                int(abs(lb[1]) / step_size_x2) + int(abs(coord[1]) / step_size_x2) - 1 if coord[1] > 0 else int(abs(lb[1]) / step_size_x2) - int(abs(coord[1]) / step_size_x2)]
                for coord in obs[:,:2]])

        # print(ij_obs)

        # The indices are then set to 1 in the blank arena size matrix.
        blank_image[ij_obs[:,0], ij_obs[:,1]] = 1

        return blank_image

    def calculate_entropy(self, state):
        """
        input:
        -state: Array containing Y_mean and Y_std from a gp predicting the functional space.

        output:
        -entropy_: Shannon's information entropy calculation using the variance.
        """
        Y_mean, Y_std = state[0], state[1]
        variance = np.square(Y_std)

        # probabilities = 1 / (Y_std * np.sqrt(2 * np.pi))
        # entropy_ = entropy(probabilities.flatten())

        entropy_ = 1 / 2 * (np.log(2 * np.pi * variance) + 1)

        # print("entropy", entropy_)

        return entropy_.flatten()

    def calculate_mi(self):
        reduced_set = self.pca.fit_transform(self.D_old[:,:2])
        return mutual_info_regression(X=reduced_set.reshape(-1, 1), y=self.D_old[:,2])[0]

    def render(self, mode='human'):
        pass

    def close (self):
        quit()