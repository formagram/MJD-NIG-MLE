import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.special import k1

def NIG_sim(params: tuple, steps: int, dt: float) -> np.ndarray:
   mu, alpha, beta, delta = params
   gamma = np.sqrt(alpha**2 - beta**2)
   IG = np.random.wald(delta * dt / gamma, (delta * dt) ** 2, steps)
   dW = np.random.normal(0, 1, steps)
   X_t = mu * dt + beta * IG + dW * np.sqrt(IG)
   return X_t

def NIG_density(x: np.ndarray, params: tuple, dt: float) -> np.ndarray:
   mu, alpha, beta, delta = params
   mu = dt * mu
   delta = dt * delta
   gamma = np.sqrt(alpha**2 - beta**2)
   arg = np.sqrt(delta**2 + np.power(x - mu, 2))
   numerator = alpha * delta * k1(alpha * arg)
   denominator = np.pi * arg
   exponent = np.exp(delta * gamma + beta * (x - mu))
   density = (numerator / denominator) * exponent
   return np.maximum(density, 1e-30)

# Set a random seed for reproducibility
np.random.seed(1)

# Simulation parameters
dt = 1/252
T = 3
N_steps = int(T / dt)

# True parameters
mu = 0.05
alpha = 60
beta = -5
delta = 0.3 ** 2
true_params = (mu, alpha, beta, delta)

# Sample path simulation
n_paths = 3
time = np.linspace(0, T, N_steps)
colors = [ "grey", "darkblue", "black"]
linestyles = [ ":", "-", "--"]

plt.figure(figsize=(9, 6))
for i in range(len(colors)):
    log_returns = NIG_sim(true_params, N_steps, dt)
    S_t = 100 * np.exp(np.cumsum(log_returns))
    plt.plot(time, S_t, lw=1, color=colors[i])

plt.title("Sample Paths", fontsize=12)
plt.xlabel("Time (Years)", fontsize=10)
plt.ylabel("Price", fontsize=10)
plt.grid(True, linestyle=':', linewidth=0.5)
plt.tight_layout()
plt.show()

# Reinitialize parameters 
mu = 0.05
alpha = 200
beta = -5
delta = 0.5
true_params = (mu, alpha, beta, delta)

# x-axis for density plots
x = np.linspace(-0.05, 0.05, 1000)

# Parameter variations
alphas = [10, 200, 500]
betas = [-150, 0, 150]
deltas = [0.1, 0.25, 0.5]
mus = [-0.75, 0, 0.75]

fig, axs = plt.subplots(2, 2, figsize=(12, 8))

# Top Left: Varying α
for i, alpha_val in enumerate(alphas):
    density = NIG_density(x, (mu, alpha_val, beta, delta), dt)
    axs[0, 0].plot(x, density, color=colors[i], label=f'α = {alpha_val}', linestyle=linestyles[i])
axs[0, 0].set_title("Varying α")
axs[0, 0].set_xlabel("Log Return")
axs[0, 0].set_ylabel("Density")
axs[0, 0].legend()
axs[0, 0].grid(True, linestyle=':', linewidth=0.5)
axs[0, 0].set_xticks(np.arange(-0.05, 0.0501, 0.01))

# Top Right: Varying β
for i, beta_val in enumerate(betas):
    density = NIG_density(x, (mu, alpha, beta_val, delta), dt)
    axs[0, 1].plot(x, density, color=colors[i], label=f'β = {beta_val}', linestyle=linestyles[i])
axs[0, 1].set_title("Varying β")
axs[0, 1].set_xlabel("Log Return")
axs[0, 1].set_ylabel("Density")
axs[0, 1].legend()
axs[0, 1].grid(True, linestyle=':', linewidth=0.5)
axs[0, 1].set_xticks(np.arange(-0.05, 0.0501, 0.01))

# Bottom Left: Varying δ
for i, delta_val in enumerate(deltas):
    density = NIG_density(x, (mu, alpha, beta, delta_val), dt)
    axs[1, 0].plot(x, density, color=colors[i], label=f'δ = {delta_val}', linestyle=linestyles[i])
axs[1, 0].set_title("Varying δ")
axs[1, 0].set_xlabel("Log Return")
axs[1, 0].set_ylabel("Density")
axs[1, 0].legend()
axs[1, 0].grid(True, linestyle=':', linewidth=0.5)
axs[1, 0].set_xticks(np.arange(-0.05, 0.0501, 0.01))

# Bottom Right: Varying μ
for i, mu_val in enumerate(mus):
    density = NIG_density(x, (mu_val, alpha, beta, delta), dt)
    axs[1, 1].plot(x, density, color=colors[i], label=f'μ = {mu_val}', linestyle=linestyles[i])
axs[1, 1].set_title("Varying μ")
axs[1, 1].set_xlabel("Log Return")
axs[1, 1].set_ylabel("Density")
axs[1, 1].legend()
axs[1, 1].grid(True, linestyle=':', linewidth=0.5)
axs[1, 1].set_xticks(np.arange(-0.05, 0.0501, 0.01))

plt.tight_layout()
plt.show()
