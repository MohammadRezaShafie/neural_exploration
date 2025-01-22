import numpy as np
import itertools
import random
import torch
from neural_exploration import *
class WheelContextualBandit(ContextualBandit):
    def __init__(self, T, delta, noise_std=0.01, seed=None):
        # In this wheel bandit the reward function h is not used
        # because rewards are generated with custom logic.
        self.delta = delta  # Store the exploration parameter.
        self.u1 = 1.2
        self.u2 = 1.0
        self.u3 = 50
        # Call parent constructor with dummy h (lambda returns 0) because we override rewards.
        super().__init__(T=T, n_arms=5, n_features=2, h=lambda x: 0.0, noise_std=noise_std, seed=seed)
        

    def reset_features(self):
        # Generate a (T, 2) array of contexts uniformly drawn from [-1, 1]
        contexts = np.random.uniform(-1, 1, (self.T, 2))
        # Replicate the context for each of the 5 arms to obtain shape (T, 5, 2)
        self.features = np.repeat(contexts[:, np.newaxis, :], self.n_arms, axis=1)
        # Also store the unique context per round for reward generation convenience.
        self.contexts = contexts

    def reset_rewards(self):
        
        self.rewards = np.zeros((self.T, self.n_arms))  # Initialize rewards

        for t in range(self.T):
            context = self.contexts[t]
            norm_X = np.linalg.norm(context)
            print(f"{context=}")
            print(f"{norm_X=}")
            if norm_X <= self.delta:
                # Inside the blue circle: assign rewards for all arms.
                self.rewards[t, 0] = np.random.normal(self.u1, self.noise_std)
                self.rewards[t, 1] = np.random.normal(self.u2, self.noise_std)
                self.rewards[t, 2] = np.random.normal(self.u2, self.noise_std)
                self.rewards[t, 3] = np.random.normal(self.u2, self.noise_std)
                self.rewards[t, 4] = np.random.normal(self.u2, self.noise_std)
            else:
                # Outside the blue circle.
                # Assign a baseline for arm 0.
                self.rewards[t, 0] = np.random.normal(self.u1, self.noise_std)
                # Depending on context signs, assign reward to one arm.
                X1, X2 = context
                if X1 > 0 and X2 > 0:
                    self.rewards[t, 1] = np.random.normal(self.u3, self.noise_std)
                    self.rewards[t, 2] = np.random.normal(self.u2, self.noise_std)
                    self.rewards[t, 3] = np.random.normal(self.u2, self.noise_std)
                    self.rewards[t, 4] = np.random.normal(self.u2, self.noise_std)
                elif X1 > 0 and X2 < 0:
                    self.rewards[t, 2] = np.random.normal(self.u3, self.noise_std)
                    self.rewards[t, 1] = np.random.normal(self.u2, self.noise_std)
                    self.rewards[t, 3] = np.random.normal(self.u2, self.noise_std)
                    self.rewards[t, 4] = np.random.normal(self.u2, self.noise_std)
                elif X1 < 0 and X2 > 0:
                    self.rewards[t, 3] = np.random.normal(self.u3, self.noise_std)
                    self.rewards[t, 1] = np.random.normal(self.u2, self.noise_std)
                    self.rewards[t, 2] = np.random.normal(self.u2, self.noise_std)
                    self.rewards[t, 4] = np.random.normal(self.u2, self.noise_std)
                else:
                    self.rewards[t, 4] = np.random.normal(self.u3, self.noise_std)
                    self.rewards[t, 2] = np.random.normal(self.u2, self.noise_std)
                    self.rewards[t, 3] = np.random.normal(self.u2, self.noise_std)
                    self.rewards[t, 1] = np.random.normal(self.u2, self.noise_std)
                # The remaining arms (if not explicitly assigned) will stay at 0.
            print(f"{self.rewards[t]=}")
        # For compatibility with the UCB code, we precompute the oracle best rewards and actions.
        self.best_rewards_oracle = np.max(self.rewards, axis=1)
        self.best_actions_oracle = np.argmax(self.rewards, axis=1)

    def reset(self):
        """
        Overriding reset to refresh both features and rewards.
        """
        self.reset_features()
        self.reset_rewards()
