import numpy as np
import random

class WheelBandit:
    def __init__(self, T, delta, noise_std=0.01, seed=None):
        """
        Initialize the WheelBandit instance.
        
        :param T: Number of rounds
        :param delta: Exploration parameter (blue circle radius)
        :param noise_std: Standard deviation of the reward noise
        :param seed: Random seed for reproducibility
        """
        self._seed(seed)
        self.T = T  # number of rounds
        self.delta = delta  # exploration parameter
        self.noise_std = noise_std  # reward noise
        self.k = 5  # number of arms
        self.contexts = np.random.uniform(-1, 1, (T, 2))  # 2D contexts (uniform on the unit disk)
        self.rewards = np.zeros((T, self.k))  # reward matrix for T rounds and k arms
        self._generate_rewards()

    def _seed(self, seed=None):
        """Set the random seed for reproducibility."""
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

    def _generate_rewards(self):
        """Generate rewards for each arm and each round."""
        for t in range(self.T):
            context = self.contexts[t]
            norm_X = np.linalg.norm(context)

            # If the context is inside the blue circle
            if norm_X <= self.delta:
                # Action 1 has mean 1.2
                self.rewards[t, 0] = np.random.normal(1.2, self.noise_std)
                # Action 2 has mean 1.0
                self.rewards[t, 1] = np.random.normal(1.0, self.noise_std)
                # Action 3 has mean 50.0 (optimal in this region)
                self.rewards[t, 2] = np.random.normal(50.0, self.noise_std)
                # Actions 4 and 5 have random sub-optimal rewards
                self.rewards[t, 3] = np.random.normal(10.0, self.noise_std)
                self.rewards[t, 4] = np.random.normal(5.0, self.noise_std)
            else:
                # If the context is outside the blue circle, reward depends on context sign
                X1, X2 = context
                if X1 > 0 and X2 > 0:
                    self.rewards[t, 2] = np.random.normal(50.0, self.noise_std)  # Action 3 is optimal
                elif X1 > 0 and X2 < 0:
                    self.rewards[t, 3] = np.random.normal(20.0, self.noise_std)  # Action 4 is optimal
                elif X1 < 0 and X2 > 0:
                    self.rewards[t, 4] = np.random.normal(30.0, self.noise_std)  # Action 5 is optimal
                else:
                    self.rewards[t, 1] = np.random.normal(1.0, self.noise_std)  # Action 2 is optimal
                # Action 1 is sub-optimal outside the blue circle
                self.rewards[t, 0] = np.random.normal(1.2, self.noise_std)

    def get_context(self, t):
        """Return the context for round t."""
        return self.contexts[t]

    def get_reward(self, t, arm):
        """Return the reward for a given round t and arm."""
        return self.rewards[t, arm]
    
    def reset(self):
        """Reset the bandit for a new set of rounds."""
        self.contexts = np.random.uniform(-1, 1, (self.T, 2))  # new contexts
        self.rewards = np.zeros((self.T, self.k))  # reset rewards
        self._generate_rewards()  # generate new rewards

