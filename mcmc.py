import numpy as np

def metropolis_hastings(twice_neg_log_posterior, n_samples, theta_init, proposal_std=0.1):
    """Generate samples from a posterior distribution, using the Metropolis-Hastings algorithm.

    Args:
        twice_neg_log_posterior (callable): -2*ln(p(theta|x)), takes theta (array) as input.
        n_samples (int): Number of samples per chain.
        theta_init (List[float]): Initial starting point in parameter space.
        proposal_std (float, List[float]): Standard deviation of Gaussian proposal distribution.

    Returns:
        samples (List[List[float]]): 0th dimension are samples, 1st dimension the individial parameters.
    """
    samples = [theta_init]
    current = theta_init

    for _ in range(n_samples):
        # Propose new candidate from na normal proposal distribution
        proposal = np.random.normal(loc=current, scale=proposal_std, size=2)

        # Compute acceptance ratio (in neg log space)
        twice_neg_log_current = twice_neg_log_posterior(current)
        twice_neg_log_proposal = twice_neg_log_posterior(proposal)
        twice_neg_log_alpha = max(0, twice_neg_log_proposal - twice_neg_log_current)

        # Accept/reject step
        u = np.random.rand()
        if -2*np.log(u) > twice_neg_log_alpha:
            current = proposal  # Accept proposal

        samples.append(current)

    return np.array(samples)