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
import mc_plot


class McmcProposal(object):
    def __init__(self):
        pass

    def update_proposal_cov(self):
        raise NotImplementedError

    def sample_proposal(self, n_samples=1):
        raise NotImplementedError

    def prob_ratio(self):
        raise NotImplementedError


class GaussianProposal(McmcProposal):
    def __init__(self, mu=None, cov=None):
        """!
        @brief Init
        @param mu  np_1darray
        @param cov np_ndarray. covariance matrix
        """
        self._mu = mu
        self._cov = cov
        super(GaussianProposal, self).__init__()

    def update_proposal_cov(self, past_samples):
        """!
        @brief fit gaussian proposal to past samples vector
        """
        self._cov = np.cov(past_samples.T)
        if not self._cov.shape:
            self._cov = np.reshape(self._cov, (1, 1))
        print("New proposal cov = %s" % str(self._cov))

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
    @brief Markov Chain Monte Carlo (MCMC) base class.
    Contains common methods for dealing with the likelyhood function
    and for obtaining parameter estimates from the mcmc chain.
    """
    def __init__(self, log_like_fn, proposal, **proposal_kwargs):
        """!
        @brief setup mcmc sampler
        @param log_like_fn callable.  must return log likelyhood (float)
        @param proposal string.
        @param proposal_kwargs dict.
        """
        self.log_like_fn = log_like_fn
        if proposal == 'Gauss':
            self.mcmc_proposal = GaussianProposal(proposal_kwargs)
        else:
            raise RuntimeError("ERROR: proposal type: %s not supported." % str(proposal))
        self.chain = None
        self.n_accepted = 1
        self.n_rejected = 0
        self.chain = None

    def _freeze_like_fn(self, **kwargs):
        """!
        @brief Freezes the likelyhood function.
        the log_like_fn should have signature:
            self.log_like_fn(theta, data=np.array([...]), **kwargs)
            and must return the log likelyhood
        """
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
        """! @brief Run the mcmc_chain.  Must be overridden."""
        raise NotImplementedError

    @property
    def acceptance_fraction(self):
        """!
        @brief Ratio of accepted samples vs total samples
        """
        return self.n_accepted / (self.n_accepted + self.n_rejected)


def mh_kernel(i, mcmc_sampler, theta_chain):
    """!
    @brief Metropolis-Hastings mcmc kernel.
    Kernel, \f[K\f] maps the pevious chain state to the new chain state:
    \f[
    K x = x'
    \f]
    For MH:
    \f[
    K(x->x') =
    \f]
    In order for repeated application of an MCMC kernel to converge
    to the correct posterior distribution \f[\Pi()\f],
    it is sufficient but not neccissary to obey detailed balance:
    \f[
    P(x^i|x^{i-1})P(x^{i}) = P(x^{i-1}|x^{i})P(x^{i-1})
    \f]
    or
    \f[
    x_i K_{ij} = x_j K_{ji}
    \f]
    This means that a step in the chain is reversible.
    The goal in MCMC is to find \f[ K \f] that makes the state vector x \f[ x \f]
    become stationary at the desired probability distribution \f[ \Pi() \f]
    @param i  int. Current chain index
    @param mcmc_sampler McmcSampler instance
    @param theta_chain  np_ndarray for sample storage
    """
    theta = theta_chain[i, :]
    # set the gaussian proposal to be centered at current loc
    mcmc_sampler.mcmc_proposal.mu = theta
    # gen random test value
    a_test = np.random.uniform(0, 1, size=1)
    # propose a new place to go
    theta_prop = mcmc_sampler.mcmc_proposal.sample_proposal()
    # compute acceptance ratio
    a_ratio = np.min((1, mcmc_sampler.mcmc_proposal.prob_ratio(
        mcmc_sampler.like_fn,
        theta,
        theta_prop)))
    if a_ratio > 1.:
        # accept proposal, it is in area of higher prob density
        theta_chain[i+1, :] = theta_prop
        mcmc_sampler.n_accepted += 1
    elif a_test < a_ratio:
        # accept proposal, even though it is "worse"
        theta_chain[i+1, :] = theta_prop
        mcmc_sampler.n_accepted += 1
    else:
        # stay put, reject proposal
        theta_chain[i+1, :] = theta
        mcmc_sampler.n_rejected += 1
    return theta_chain


class Metropolis(McmcSampler):
    """!
    @brief Metropolis Markov Chain Monte Carlo (MCMC) sampler.
    Proposal distribution is gaussian and symetric
    """
    def __init__(self, log_like_fn, **proposal_kwargs):
        proposal = 'Gauss'
        super(Metropolis, self).__init__(log_like_fn, proposal, **proposal_kwargs)

    def _mcmc_run(self, n, theta_0, cov_est=5.0, **kwargs):
        """!
        @brief Run the metropolis algorithm.
        @param n  int. number of samples to draw.
        @param theta_0 np_1darray. initial guess for parameters.
        @param cov_est float or np_1darray.  Initial guess of anticipated theta variance.
            strongly recommended to specify, but is optional.
        """
        self._freeze_like_fn(**kwargs)
        # pre alloc storage for solution
        self.n_accepted = 1
        self.n_rejected = 0
        theta_chain = np.zeros((n, np.size(theta_0)))
        self.chain = theta_chain
        theta_chain[0, :] = theta_0
        self.mcmc_proposal.cov = np.eye(len(theta_0)) * cov_est
        for i in range(n - 1):
            # M-H Kernel
            mh_kernel(i, self, theta_chain)
        self.chain = theta_chain


class AdaptiveMetropolis(McmcSampler):
    """!
    @brief Metropolis Markov Chain Monte Carlo (MCMC) sampler.
    """
    def __init__(self, log_like_fn, **proposal_kwargs):
        proposal = 'Gauss'
        super(AdaptiveMetropolis, self).__init__(log_like_fn, proposal, **proposal_kwargs)

    def _mcmc_run(self, n, theta_0, cov_est=5.0, **kwargs):
        """!
        @brief Run the metropolis algorithm.
        @param n  int. number of samples to draw.
        @param theta_0 np_1darray. initial guess for parameters.
        @param cov_est float or np_1darray.  Initial guess of anticipated theta variance.
            strongly recommended to specify, but is optional.
        @param adapt int.  Sample index at which to begin adaptively updating the
            proposal distribution (default == 200)
        @param lag  int.  Number of previous samples to use for proposal update (default == 100)
        @param lag_mod.  Number of iterations to wait between updates (default == 1)
        """
        adapt = kwargs.get("adapt", 200)
        lag = kwargs.get("lag", 100)
        lag_mod = kwargs.get("lag_mod", 10)
        self._freeze_like_fn(**kwargs)
        # pre alloc storage for solution
        self.n_accepted = 1
        self.n_rejected = 0
        theta_chain = np.zeros((n, np.size(theta_0)))
        self.chain = theta_chain
        theta_chain[0, :] = theta_0
        self.mcmc_proposal.cov = np.eye(len(theta_0)) * cov_est
        for i in range(n - 1):
            # M-H Kernel
            mh_kernel(i, self, theta_chain)
            # continuously update the proposal distribution
            if (lag > adapt):
                raise RuntimeError("lag must be smaller than adaptation start index")
            if i >= adapt and (i % lag_mod) == 0:
                print("  Updating proposal cov at sample index = %d" % i)
                current_chain = theta_chain[:i, :]
                self.mcmc_proposal.update_proposal_cov(current_chain[-lag:, :])
        self.chain = theta_chain


if __name__ == "__main__":
    ## EXAMPLE 1: Sample from a gaussian distribution
    mu_gold, std_dev_gold = 5.0, 0.5

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
    my_mcmc = AdaptiveMetropolis(log_like_fn)
    my_mcmc.run_mcmc(1000, theta_0, data=[0, 0, 0], cov_est=5.0)
    # view results
    theta_est, sig_est, chain = my_mcmc.param_est(200)
    print("Esimated mu: %f" % theta_est)
    print("Estimated sigma: %f " % sig_est)
    print("Acceptance fraction: %f" % my_mcmc.acceptance_fraction)
    # vis the parameter estimates
    mc_plot.plot_mcmc_params(chain, ["$\mu$"], savefig='gauss_mu_mcmc_ex.png', truths=[5.0])
    # vis the full chain
    theta_est_, sig_est_, full_chain = my_mcmc.param_est(0)
    mc_plot.plot_mcmc_chain(full_chain, ["$\mu$"], savefig='gauss_mu_chain_ex.png', truths=[5.0])

    ## EXAMPLE 2: Fit a line to data
