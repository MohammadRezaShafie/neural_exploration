import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from .ucb import UCB
from .utils import inv_sherman_morrison

class PolyLinUCB(UCB):
    """
    Polynomial (Quadratic) LinUCB.
    Applies a degree-2 polynomial transform on the original context 
    before performing the standard linear UCB steps.
    """

    def __init__(
        self,
        bandit,
        degree=2,
        interaction_only=False,
        include_bias=False,
        reg_factor=1.0,
        delta=0.01,
        bound_theta=1.0,
        confidence_scaling_factor=0.0,
        throttle=int(1e2),
    ):
        """
        Parameters
        ----------
        bandit : object
            The bandit problem instance (must have .features, .rewards, etc.).
        degree : int
            Polynomial degree to expand features. (Default=2 for quadratic)
        interaction_only : bool
            If True, only use interaction features (no x^2 terms).
        include_bias : bool
            Whether to include a bias column in the transform.
        ...
        """
        
        self.degree = degree
        self.bound_theta = bound_theta
        # Precompute the transform dimension
        self.poly = PolynomialFeatures(
            degree=self.degree,
            interaction_only=interaction_only,
            include_bias=include_bias
        )
        
        # We do a "dummy" transform on a single sample to see how many dimensions result
        dummy_x = np.zeros((1, bandit.n_features))
        self._poly_dim = self.poly.fit_transform(dummy_x).shape[1]
        super().__init__(
            bandit,
            reg_factor=reg_factor,
            confidence_scaling_factor=confidence_scaling_factor,
            delta=delta,
            throttle=throttle,
        )


    def reset(self):
        """Re-initialize tracking variables."""
        self.reset_upper_confidence_bounds()
        self.reset_regrets()
        self.reset_actions()
        self.reset_A_inv()
        self.reset_grad_approx()
        self.iteration = 0

        # Randomly init polynomial-space theta
        self.theta = np.random.uniform(
            low=-1.0,
            high=1.0,
            size=(self.bandit.n_arms, self._poly_dim)
        ) * self.bound_theta

        # Weighted sum of features in polynomial space
        self.b = np.zeros((self.bandit.n_arms, self._poly_dim))

    @property
    def approximator_dim(self):
        """Dimension of the polynomially expanded features."""
        return self._poly_dim

    @property
    def confidence_multiplier(self):
        """Same formula as standard LinUCB or you can customize."""
        return (
            self.confidence_scaling_factor *
            np.sqrt(
                self.approximator_dim *
                np.log(
                    1 + self.iteration * (self.bound_theta ** 2)
                    / (self.reg_factor * self.approximator_dim)
                ) + 2*np.log(1/self.delta)
            ) +
            np.sqrt(self.reg_factor)*self.bound_theta
        )

    def update_output_gradient(self):
        """
        For linear approximators with polynomial expansions:
        gradient wrt theta = the polynomially expanded features themselves.
        """
        # bandit.features[t] is shape (n_arms, d)
        # We transform each arm’s feature to polynomial space.
        features_t = self.bandit.features[self.iteration]  # shape (n_arms, d)
        poly_features_t = self.poly.transform(features_t)   # shape (n_arms, poly_dim)
        self.grad_approx = poly_features_t

    def train(self):
        """
        Standard LinUCB parameter update in polynomial space.
        We add the new (feature * reward) into b for the chosen arm.
        Then update theta using A_inv.
        """
        a = self.action
        t = self.iteration

        # transform the chosen arm's features
        feats_ta = self.bandit.features[t, a].reshape(1, -1)
        poly_feats_ta = self.poly.transform(feats_ta).ravel()  # shape (poly_dim,)
        
        reward_ta = self.bandit.rewards[t, a]
        self.b[a] += poly_feats_ta * reward_ta

        # update theta
        self.theta = np.array([
            np.matmul(self.A_inv[arm_idx], self.b[arm_idx])
            for arm_idx in self.bandit.arms
        ])

    def predict(self):
        """
        Predicted reward for each arm = theta[arm] dot poly_features[arm].
        """
        t = self.iteration

        # We call update_output_gradient first from update_confidence_bounds().
        # So self.grad_approx is already the polynomial expansion 
        # for all arms at time t: shape (n_arms, poly_dim).
        self.mu_hat[t] = np.array([
            np.dot(self.grad_approx[arm], self.theta[arm])
            for arm in self.bandit.arms
        ])
