from sklearn import gaussian_process
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.utils.optimize import _check_optimize_result
from sklearn.feature_selection import mutual_info_regression
import scipy.optimize as opt

from warnings import simplefilter
from sklearn.exceptions import ConvergenceWarning

simplefilter("ignore", category=ConvergenceWarning)


class CustomGP(GaussianProcessRegressor):
    def __init__(self, *args, max_iter=5e05, gtol=1e-06, resolution=100, **kwargs):
        super().__init__(*args, **kwargs)
        self._max_iter = max_iter
        self._gtol = gtol
        self.resolution = resolution

    def _constrained_optimization(self, obj_func, initial_theta, bounds):
        if self.optimizer == "fmin_l_bfgs_b":
            opt_res = opt.minimize(obj_func, initial_theta, method="L-BFGS-B", jac=True, bounds=bounds, options={'maxiter':self._max_iter, 'gtol': self._gtol})
            _check_optimize_result("lbfgs", opt_res)
            theta_opt, func_min = opt_res.x, opt_res.fun
        elif callable(self.optimizer):
            theta_opt, func_min = self.optimizer(obj_func, initial_theta, bounds=bounds)
        else:
            raise ValueError("Unknown optimizer %s." % self.optimizer)
        return theta_opt, func_min

    def _fit_gp(self, D_old, X_true):
        """Returns predicted output from the functional space after fitted with D_old."""
        self.fit(D_old[:,:2], D_old[:,2])
        Y_mean, Y_std = self.predict(X_true, return_std=True)
        return Y_mean.reshape(self.resolution, -1), Y_std.reshape(self.resolution, -1)