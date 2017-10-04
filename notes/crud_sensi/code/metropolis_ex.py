###
# author: William Gurecky
# date: Oct 2 2017
# license: MIT
# notes:  Implementation of the Metropolis MCMC algo for educational purposes.
###
from __future__ import print_function, division
from six import iteritems
import numpy as np
import scipy.stats as stats


class McmcKernel(object):
    def __init__(self):
        pass

    def sample_proposal(self, n_samples=1):
        raise NotImplementedError

    def prob_ratio(self):
        raise NotImplementedError


class GaussianKernel(McmcKernel):
    def __init__(self, mu=None, cov=None):
        """!
        @brief Init
        @param mu  np_1darray
        @param cov np_ndarray. covariance matrix
        """
        self._mu = mu
        self._cov = cov
        super(GaussianKernel, self).__init__()

    def update_kernel(self, past_samples):
        """!
        @brief fit gaussian kernel to past samples vector
        """
        # self._mu = np.mean(past_samples, axis=0)
        self._cov = np.cov(past_samples.T)

    def sample_proposal(self, n_samples=1):
        """!
        @brief Sample_proposal distribution
        """
        mu = self.mu
        cov = self.cov
        return np.random.multivariate_normal(mu, cov, size=n_samples)

    def prob_ratio(self, like_fn, theta_past, theta_proposed):
        """!
        @brief evaluate probability ratio:
        \f[
        \frac{\Pi(x^i)}{\Pi(x^{i-1})} \frac{g(x^{i-1}|x^i)}{g(x^i| x^{i-1})}
        \f]
        Where \f$ g() \f$ is the proposal distribution fn
        and \f[ \Pi \f] is the likelyhood function
        """
        qx = np.linspace(0, 5, 10, endpoint=False)
        x = np.ones((len(qx), len(theta_past))) * qx
        g_ratio = lambda x_0, x_1: \
            stats.multivariate_normal.pdf(x_0, mean=theta_past, cov=self.cov) / \
            stats.multivariate_normal.pdf(x_1, mean=theta_proposed, cov=self.cov)
        g_r = g_ratio(theta_proposed, theta_past)  # should be 1 in symmetric case
        past_likelihood = like_fn(theta_past)
        if past_likelihood <= 0:
            # will result in div by zero error. Dont step here!
            return 0.0
        proposed_likelihood = like_fn(theta_proposed)
        return (proposed_likelihood / past_likelihood) * g_r

    @property
    def mu(self):
        return self._mu

    @mu.setter
    def mu(self, mu):
        self._mu = mu

    @property
    def cov(self):
        return self._cov

    @cov.setter
    def cov(self, cov):
        self._cov = cov


class McmcSampler(object):
    """!
    @brief Markov Chain Monte Carlo (MCMC) generic sampler
    """
    def __init__(self, log_like_fn, kernel, **kernel_kwargs):
        """!
        @brief setup mcmc sampler
        @param log_like_fn callable.  must return log likelyhood (float)
        @param kernel string.
        @param kernel_kwargs dict.
        """
        self.log_like_fn = log_like_fn
        if kernel == 'Gauss':
            self.mcmc_kernel = GaussianKernel(kernel_kwargs)
        else:
            raise RuntimeError("ERROR: kernel type: %s not supported." % str(kernel))
        self.chain = None
        self.n_accepted = 1
        self.n_rejected = 0
        self.chain = None

    def _freeze_like_fn(self, **kwargs):
        self.like_fn = lambda theta: np.exp(self.log_like_fn(theta, **kwargs))

    def param_est(self, n_burn):
        """!
        @brief Computes mean an std of sample chain discarding the first n_burn samples.
        @return  mean (np_1darray), std (np_1darray), chain (np_ndarray)
        """
        chain_slice = self.chain[n_burn:, :]
        mean_theta = np.mean(chain_slice, axis=0)
        std_theta = np.std(chain_slice.T)
        return mean_theta, std_theta, chain_slice

    def run_mcmc(self, n, theta_0, **kwargs):
        """!
        @brief Run the mcmc chain
        @param n int.  number of samples in chain to draw
        @param theta_0 np_1darray of initial guess
        """
        self._mcmc_run(n, theta_0, **kwargs)

    def _mcmc_run(self, *args, **kwarags):
        raise NotImplementedError

    @property
    def acceptance_fraction(self):
        """!
        @brief Ratio of accepted samples vs total samples
        """
        return self.n_accepted / (self.n_accepted + self.n_rejected)


class Metropolis(McmcSampler):
    """!
    @brief Metropolis Markov Chain Monte Carlo (MCMC) generic sampler.
    Proposal distribution is gaussian and symetric
    """
    def __init__(self, log_like_fn, **kernel_kwargs):
        kernel = 'Gauss'
        super(Metropolis, self).__init__(log_like_fn, kernel, **kernel_kwargs)

    def _mcmc_run(self, n, theta_0, cov_est=5.0, **kwargs):
        """!
        @brief Run the metropolis algorithm.
        @param cov_est float or np_1darray of anticipated theta variance
        """
        self._freeze_like_fn(**kwargs)
        # pre alloc storage for solution
        self.n_accepted = 1
        self.n_rejected = 0
        theta_chain = np.zeros((n, np.size(theta_0)))
        self.chain = theta_chain
        theta_chain[0, :] = theta_0
        self.mcmc_kernel.cov = np.eye(len(theta_0)) * cov_est
        for i in range(n - 1):
            theta = theta_chain[i, :]
            # set the gaussian kernel to be centered at current loc
            self.mcmc_kernel.mu = theta
            # gen random test value
            a_test = np.random.uniform(0, 1, size=1)
            # propose a new place to go
            theta_prop = self.mcmc_kernel.sample_proposal()
            # compute acceptance ratio
            a_ratio = np.min((1, self.mcmc_kernel.prob_ratio(
                self.like_fn,
                theta,
                theta_prop)))
            if a_ratio > 1.:
                # accept proposal, it is in area of higher prob density
                theta_chain[i+1, :] = theta_prop
                self.n_accepted += 1
            elif a_test < a_ratio:
                # accept proposal, even though it is "worse"
                theta_chain[i+1, :] = theta_prop
                self.n_accepted += 1
            else:
                # stay put, reject proposal
                theta_chain[i+1, :] = theta
                self.n_rejected += 1
        self.chain = theta_chain


if __name__ == "__main__":
    """! @brief simple mcmc gaussian sample test. """
    mu_gold, std_dev_gold = 5.0, 1.5

    def log_like_fn(theta, data=None):
        return np.log(stats.norm.pdf(theta[0],
                                     loc=mu_gold,
                                     scale=std_dev_gold)) - log_prior(theta)

    def log_prior(theta):
        if -100 < theta[0] < 100:
            return 0
        else:
            return -np.inf

    # init sampler
    theta_0 = np.array([1.0])
    my_mcmc = Metropolis(log_like_fn)
    my_mcmc.run_mcmc(500, theta_0, data=[0, 0, 0])
    # view results
    theta_est, sig_est, chain = my_mcmc.param_est(100)
    print(theta_est)
    print(sig_est)
    print("Acceptance fraction: %f" % my_mcmc.acceptance_fraction)
    # vis the chain
    # vis the parameter estimates
