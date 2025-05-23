import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from math import factorial


def MJD_sim(params: tuple, T: int, dt: float) -> np.ndarray:

    # Unpack parameters
    mu, sig, lam, muJ, sigJ = params

    # Number of time steps
    steps = int(T * (1/dt))
    
    # Compound Poisson Process: simulate number of jumps and jump sizes
    dN_t = np.random.poisson(lam=lam * dt, size=steps)
    Y_t  = np.random.lognormal(mean=muJ, sigma=sigJ, size=steps)
    dJ_t = dN_t * np.log(Y_t)
    
    # Jump compensation
    k = np.exp(muJ + (sigJ**2) / 2) - 1 
    
    # Wiener Process (Brownian motion)
    dW_t = np.random.normal(loc=0, scale=np.sqrt(dt), size=steps)
    
    # Log returns combining drift, diffusion, and jump components
    log_returns = (mu - sig**2 / 2 - lam * k) * dt + sig * dW_t + dJ_t
    
    return log_returns

def MJD_density(x: np.ndarray, params: tuple, dt: float) -> np.ndarray:
    
    # Unpack parameters
    mu, sig, lam, muJ, sigJ = params

    # Init density
    density = np.zeros_like(x)

    # Jump compensation term
    k = np.exp(muJ + (sigJ**2) / 2) - 1 

    # Sum the contributions for 0 to 19 jumps
    for n in range(20):
        mu_den = (mu - (sig**2) / 2 - lam * k) * dt + n * muJ
        var_den = (sig**2) * dt + (sigJ**2) * n
        var_den = np.maximum(var_den, 1e-30)  # avoid zero variance
        norm_den = (1 / np.sqrt(2 * np.pi * var_den)) * np.exp(-((x - mu_den)**2) / (2 * var_den))
        poisson_den = np.exp(-lam * dt) * ((lam * dt)**n) / factorial(n)
        density += poisson_den * norm_den
    return np.maximum(density, 1e-30)

np.random.seed(3)

# Base parameters
T = 3
dt = 1/252

mu_true   = 0.1    # drift
sig_true  = 0.2    # volatility
lam_true  = 10      # jump intensity
muJ_true  = -0.1   # jump mean
sigJ_true = 0.05    # jump volatility
true_params = (mu_true, sig_true, lam_true, muJ_true, sigJ_true)

# Sample path simulation
n_paths = 3
N_steps = int(T / dt)
time = np.linspace(0, T, N_steps)

# Colors
colors = [ "grey", "darkblue", "black"]
linestyles = [ ":", "-", "--"]

# --- Plot 1: Sample Paths ---
plt.figure(figsize=(9, 6))
for i in range(len(colors)):
    log_returns = MJD_sim(true_params, T, dt)
    S_t = 100 * np.exp(np.cumsum(log_returns))
    plt.plot(time, S_t, lw=1, color=colors[i])

plt.title("Sample Paths", fontsize=12)
plt.xlabel("Time (Years)", fontsize=10)
plt.ylabel("Price", fontsize=10)
plt.grid(True, linestyle=':', linewidth=0.5)
plt.tight_layout()
plt.show()

# x-axis for density plots
x = np.linspace(-0.1, 0.1, 1000)

# Parameters to vary
lambdas = [2, 10, 30]
muJs = [-0.1, 0, 0.1]
sigJs = [0.02, 0.1, 0.3]

# --- Plot 2: Densities in 2x2 layout (3 used) ---
fig, axs = plt.subplots(2, 2, figsize=(12, 8))

# Top Left: Varying λ
for i, lam_val in enumerate(lambdas):
    density = MJD_density(x, (mu_true, sig_true, lam_val, muJ_true, sigJ_true), dt)
    axs[0, 0].plot(x, density, color=colors[i], label=f'λ = {lam_val}', linestyle=linestyles[i])
axs[0, 0].set_title("Varying λ")
axs[0, 0].set_xlabel("Log Return")
axs[0, 0].set_ylabel("Density")
axs[0, 0].legend()
axs[0, 0].grid(True, linestyle=':', linewidth=0.5)
axs[0, 0].set_xticks(np.arange(-0.1, 0.11, 0.05))

# Top Right: Varying μJ
for i, muJ_val in enumerate(muJs):
    density = MJD_density(x, (mu_true, sig_true, lam_true, muJ_val, sigJ_true), dt)
    axs[0, 1].plot(x, density, color=colors[i], label=f'μj = {muJ_val}', linestyle=linestyles[i])
axs[0, 1].set_title("Varying μj")
axs[0, 1].set_xlabel("Log Return")
axs[0, 1].set_ylabel("Density")
axs[0, 1].legend()
axs[0, 1].grid(True, linestyle=':', linewidth=0.5)
axs[0, 1].set_xticks(np.arange(-0.1, 0.11, 0.05))

# Bottom Left: Varying σJ
for i, sigJ_val in enumerate(sigJs):
    density = MJD_density(x, (mu_true, sig_true, lam_true, muJ_true, sigJ_val), dt)
    axs[1, 0].plot(x, density, color=colors[i], label=f'σj = {sigJ_val}', linestyle=linestyles[i])
axs[1, 0].set_title("Varying σj")
axs[1, 0].set_xlabel("Log Return")
axs[1, 0].set_ylabel("Density")
axs[1, 0].legend()
axs[1, 0].grid(True, linestyle=':', linewidth=0.5)
axs[1, 0].set_xticks(np.arange(-0.1, 0.11, 0.05))

# Hide bottom right (unused)
axs[1, 1].axis('off')

plt.tight_layout()
plt.show()
