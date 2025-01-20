import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from neural_exploration import *
sns.set()
T = int(10000)
n_arms = 4
n_features = 20
noise_std = 0.1

confidence_scaling_factor = noise_std

n_sim = 4

SEED = 42
np.random.seed(SEED)

max_threshold=10
w_knn=0.8
alpha = 1
kappa_weight = 0.5

### mean reward function
a = np.random.randn(n_features)
A = np.random.normal(0, 1, (n_features, n_features))
a /= np.linalg.norm(a, ord=2)
reward_func_model = "h3"

if reward_func_model == "h1":
    reward_func = lambda x: 10*np.dot(a, x)**2
elif reward_func_model == "h2":
    reward_func = lambda x: np.dot(x, A.T @ A @ x)
elif reward_func_model == "h3":
    reward_func = lambda x: np.cos(3*np.dot(x, a))    


t = np.arange(T)
bandit = ContextualBandit(T, n_arms, n_features, reward_func, noise_std=noise_std, seed=SEED)


T_wheel  = 50
delta = 2
wheel_bandit = WheelBandit(T_wheel,delta= delta, noise_std=0.01, seed= SEED)


regrets = np.empty((n_sim, T))

for i in range(n_sim):
    wheel_bandit.reset()
    model = LNUCBTA(wheel_bandit,alpha=alpha, kappa_weight=kappa_weight, max_threshold=max_threshold,w_knn=w_knn)
    model.run()

    regrets[i] = np.cumsum(model.regrets)
    
# Then plot or analyze model.regrets, etc.
fig, ax = plt.subplots(figsize=(11, 4), nrows=1, ncols=1)
mean_regrets = np.mean(regrets, axis=0)
std_regrets = np.std(regrets, axis=0) / np.sqrt(regrets.shape[0])
ax.plot(mean_regrets)
ax.fill_between(t, mean_regrets - 2*std_regrets, mean_regrets + 2*std_regrets, alpha=0.15)
ax.set_title('Cumulative regret')

plt.tight_layout()

fig.savefig(f'figures/LNUCBTA_wheel_bandit_delta_{wheel_bandit}_alpha_{alpha}_maxthreshold_{max_threshold}_wknn_{w_knn}_kappa_{kappa_weight}.png')


# bandit = ContextualBandit(T, n_arms, n_features, reward_func, noise_std)

# regrets = np.empty((n_sim, T))

# for i in range(n_sim):
#     bandit.reset_rewards()
#     model = PolyLinUCB(
#         bandit,
#         degree=2,  # Quadratic expansions
#         interaction_only=False,  # include square terms x_i^2
#         include_bias=False,      # optional
#         reg_factor=1.0,
#         delta=0.01,
#         bound_theta=1.0,
#         confidence_scaling_factor=0.1,
#     )
#     model.run()

#     regrets[i] = np.cumsum(model.regrets)
    

