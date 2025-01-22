import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from .ucb import UCB
from .utils import inv_sherman_morrison


class LNUCBTA(UCB):
    """Linear UCB with Threshold-based Attention (LNUCB-TA).

    This class extends the base UCB interface and implements the
    threshold-adaptive KNN + linear predictor approach.
    """
    def __init__(
        self,
        bandit,
        alpha=1.0, 
        w_knn = 0.5,# base alpha
        min_threshold=1,            # minimum KNN threshold
        max_threshold=5,            # maximum KNN threshold
        kappa_weight=0.5,           # mixes global vs. local attention
        reg_factor=1.0,             # regularization strength (lambda)
        delta=0.01,                 # for confidence intervals
        bound_theta=1.0,            # bound for random init of theta
        confidence_scaling_factor=0.0,
        train_every=1,
        throttle=int(1e2),
    ):
        self.alpha = alpha
        self.min_threshold = min_threshold
        self.max_threshold = max_threshold
        self.kappa_weight = kappa_weight
        self.w_knn = w_knn
        # For random initialization of theta
        self.bound_theta = bound_theta
        # Maximum L2 norm of features across all arms/rounds (used in confidence bounds)
        self.bound_features = np.max(np.linalg.norm(bandit.features, ord=2, axis=-1))
        # Global average of all rewards (used in local/global attention mix)
        self.overall_avg_reward = np.mean(bandit.rewards)

        super().__init__(
            bandit=bandit,
            reg_factor=reg_factor,
            confidence_scaling_factor=confidence_scaling_factor,
            delta=delta,
            train_every=train_every,
            throttle=throttle
        )

    def reset(self):
        """Initialize (or re-initialize) all internal variables and caches."""
        self.reset_upper_confidence_bounds()
        self.reset_regrets()
        self.reset_actions()
        self.reset_A_inv()
        self.reset_grad_approx()
        self.iteration = 0

        # Randomly initialize linear predictor
        self.theta = np.random.uniform(
            low=-1.0, high=1.0,
            size=(self.bandit.n_arms, self.bandit.n_features)
        ) * self.bound_theta

        # Reward-weighted features sum per arm
        self.b = np.zeros((self.bandit.n_arms, self.bandit.n_features))

        # Count of how many times each arm is chosen
        self.N = np.zeros(self.bandit.n_arms)

        # Per-arm feature and reward history (for KNN)
        self.feature_history = [[] for _ in range(self.bandit.n_arms)]
        self.reward_history = [[] for _ in range(self.bandit.n_arms)]

        # Store local attention per arm
        self.local_attention = np.zeros(self.bandit.n_arms)

    @property
    def approximator_dim(self):
        """Dimension of the approximator parameters (same as n_features)."""
        return self.bandit.n_features

    @property
    def confidence_multiplier(self):
        """
        We'll override the default LinUCB confidence multiplier and do
        a custom, dynamic alpha below. Return 1.0 here, since we handle
        exploration scaling ourselves.
        """
        return 1.0

    def update_output_gradient(self):
        """For linear approximators, gradient wrt parameters = features."""
        self.grad_approx = self.bandit.features[self.iteration]

    def train(self):
        """
        Update parameters after taking the chosen action.
        This method is called every `train_every` rounds in UCB.run().
        """
        # Incorporate the reward from the chosen action into b
        a = self.action
        t = self.iteration

        # reward for (t, a)
        reward_ta = self.bandit.rewards[t, a]
        feats_ta = self.bandit.features[t, a]

        self.b[a] += feats_ta * reward_ta

        # Update linear predictor theta
        self.theta = np.array([
            np.matmul(self.A_inv[arm_idx], self.b[arm_idx])
            for arm_idx in self.bandit.arms
        ])

        # Update usage stats
        self.N[a] += 1
        self.feature_history[a].append(feats_ta)
        self.reward_history[a].append(reward_ta)

    def predict(self):
        """
        Base predicted reward for each arm = linear predictor + optional KNN (if threshold is met).
        We store the results in self.mu_hat[self.iteration].
        """
        t = self.iteration

        # Linear part
        linear_preds = np.array([
            np.dot(self.bandit.features[t, arm], self.theta[arm])
            for arm in self.bandit.arms
        ])

        # KNN part
        knn_preds = np.zeros(self.bandit.n_arms)
        for arm in self.bandit.arms:
            # Adaptive threshold = min_threshold + scaled variance
            var_arm = np.var(self.reward_history[arm]) if len(self.reward_history[arm]) > 1 else 0.0
            adaptive_threshold = self.min_threshold + (self.max_threshold - self.min_threshold) * var_arm

            if len(self.feature_history[arm]) >= adaptive_threshold:
                # Use distance-weighted KNN
                knn = KNeighborsRegressor(
                    n_neighbors=int(adaptive_threshold),
                    weights='distance'
                )
                knn.fit(self.feature_history[arm], self.reward_history[arm])
                # Predict for the current round's features
                feats_ta = self.bandit.features[t, arm].reshape(1, -1)
                knn_preds[arm] = knn.predict(feats_ta)[0]

        # Combine linear + KNN into mu_hat
        self.mu_hat[t] = linear_preds + knn_preds
        # self.mu_hat[t] = self.w_knn * knn_preds + (1 - self.w_knn) * linear_preds

    def update_confidence_bounds(self):
        # Step 1: gradient approx for each arm
        self.update_output_gradient()

        # Step 2: local attention
        for arm in self.bandit.arms:
            if len(self.reward_history[arm]) > 0:
                self.local_attention[arm] = np.mean(self.reward_history[arm])
            else:
                self.local_attention[arm] = 0.0

        # Step 3: dynamic alpha
        # s_arm = kappa * global_avg + (1 - kappa)* local_avg
        s = self.kappa_weight * self.overall_avg_reward + (1.0 - self.kappa_weight) * self.local_attention
        dynamic_alpha = self.alpha / (1.0 + self.N) * s

        # Step 4: exploration bonus for each arm
        bonuses = []
        for arm in self.bandit.arms:
            # sqrt(x^T A_inv x)
            x = self.grad_approx[arm]
            bonuses.append(dynamic_alpha[arm] * np.sqrt(x @ self.A_inv[arm] @ x))
        self.exploration_bonus[self.iteration] = np.array(bonuses)

        # Step 5: call predict => mu_hat[t, arm]
        self.predict()

        # Step 6: final UCB = mu_hat + bonus
        t = self.iteration
        self.upper_confidence_bounds[t] = (
            self.mu_hat[t] + self.exploration_bonus[t]
        )
