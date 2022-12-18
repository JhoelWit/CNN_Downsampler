from robot_source import Source
from sampler import Sampler

from scipy.stats.qmc import Sobol
from sklearn.decomposition import PCA
import numpy as np
from sklearn.feature_selection import mutual_info_regression

"""
This downsampler will be a baseline for the CNN.

input:
-observations: the combined new and old observations

output:
-D_filtered: the filtered observations that have a balanced level of information gain and mutual information combared to D_old
"""

class DownSampler:
    def __init__(self):
        pass

    def mi_downsample(self, data, sample_limit=100):
        """
        steps:
        1. New data is brought in and has a shape anywhwere from the sample cap to the action cap (i.e 100 to 150 observations)
        2. An optimizer will be used to find the maximum mutual information of all these observations, using the sklearn method for guassian distributions
        UPDATE: RANSAC will be used.
        3. That optimized observation set will be returned

        input:
        -new_data: This data consists of the filtered dataset from the previous step, as well as any new observations brought in by peers

        output:
        -D_filtered: The filtered data which optimizes the mutual information and the information gain. 
        """
        # dimension = 50
        # n_samples = 2**11
        # sampler = Sobol(d=dimension)
        # l_bounds = [0] * dimension
        # u_bounds = [149] * dimension
        # int_sample = sampler.integers(l_bounds=l_bounds, u_bounds=u_bounds, n=n_samples, endpoint=True)

        # unique_count = 0
        # for row in int_sample:
        #     # print(row)
        #     if len(np.unique(row)) == len(row):
        #         unique_count += 1
        # print(f"unique count is {unique_count}/{int_sample.shape[0]}")


        N = 1000

        pca = PCA(n_components=1)

        reduced_data = pca.fit_transform(data[:,:2])

        reduced_set = np.hstack((reduced_data.reshape(-1, 1), data))

        best_dict = {}
        iter = 0
        best = 0
        while iter < N:
            np.random.shuffle(reduced_set)
            temp = reduced_set[:sample_limit]

            mutual_info = mutual_info_regression(temp[:,0].reshape(-1, 1), temp[:,3])[0]
            if mutual_info > best:
                best_dict["dataset"] = temp
                best_dict["score"] = mutual_info
                best = mutual_info

            iter += 1 
        # print("best score", best)

        self.data = best_dict["dataset"][:, 1:]


        return self.data

    def bayes_downsample(self, data, active_set = 100):
        """
        steps:
        1. Take out every other sample until the data size is at the active size

        inputs:
        data: combined data to be downsampled

        outputs:
        filtered_data: downsampled data
        """

        pointer = 0
        while data.shape[0] > active_set:
            data = np.delete(data, pointer, axis=0)
            pointer = pointer - 1 if pointer < 0 else pointer + 1
            pointer = -pointer
        
        return data


if __name__ == "__main__":

    sample_cap = 100
    resolution = 100

    randomize = True
    source = Source(sample_cap, resolution, randomize)
    sampler = Sampler(sample_cap, velocity=0.1)
    source.generate_arena()

    D_old, D_new = sampler.reset(source)
    D_new = source.measure_signal(D_new)

    D_all = np.vstack((D_old, D_new))

    downsampler = DownSampler()

    D_old_filt = downsampler.downsample(data=D_all)


