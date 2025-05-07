import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
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

def MJD_init_params(data: np.ndarray, dt: float, epsilon: float) -> tuple:

    returns_no_jump = data[np.abs(data) < epsilon]
    returns_jump    = data[np.abs(data) >= epsilon]

    lambda_hat = len(returns_jump)/(len(data) * dt)

    mu_hat  = (2*np.mean(returns_no_jump) + np.var(returns_no_jump)*dt)/(2*dt)
    sig_hat = np.sqrt(np.var(returns_no_jump)/dt)

    muJ_hat = np.mean(returns_jump) - (mu_hat - (sig_hat**2)/2)*dt
    sigJ_hat = np.sqrt(np.var(returns_jump) - (sig_hat**2)*dt)

    return (mu_hat, sig_hat, lambda_hat, muJ_hat, sigJ_hat)

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

def Loglikelihood(params: np.ndarray, data: np.ndarray, dt: float) -> float:

    # Unpack parameters
    mu, sig, lam, muJ, sigJ = params

    # Compute density vector and clip too low values
    density = MJD_density(data, (mu, sig, lam, muJ, sigJ), dt)
    density = np.clip(density, 1e-10, None)

    # Compute negative log-likelihood
    log_likelihood = -np.sum(np.log(density))
    
    return  log_likelihood


def plot_parameter_distributions(estimated_params: np.ndarray, true_params: tuple, save_path="mjd_param_distributions.pdf"):  
    """
    Plot distributions of estimated MJD parameters in a publication-ready format.

    Parameters:
    - estimated_params: np.ndarray with shape (n_paths, 5)
    - true_params: tuple of true parameters (mu, sigma, lambda, muJ, sigmaJ)
    - save_path: file path to save the PDF figure
    """

    param_names = ['μ', 'σ', 'λ', 'μj', 'σj']   # Use Greek symbols for elegance
    true_values = list(true_params)

    # Create subplots: 3 rows, 2 columns (last plot will be empty)
    fig, axes = plt.subplots(3, 2, figsize=(12, 13.5))
    axes = axes.flatten()

    for i in range(5):
        axes[i].hist(estimated_params[:, i], bins=20, alpha=0.7, color='grey', edgecolor='black')
        axes[i].axvline(true_values[i], color='black', linestyle='--', linewidth=1.5, label=f'True {param_names[i]} = {true_values[i]:.4f}')
        mean_est = np.mean(estimated_params[:, i])
        axes[i].axvline(mean_est, color='darkblue', linestyle='-', linewidth=1.5, label=f'Mean = {mean_est:.4f}')
        axes[i].set_title(f'Estimated {param_names[i]} Distribution', fontsize=12)
        axes[i].set_xlabel(f'{param_names[i]} Value', fontsize=10)
        axes[i].set_ylabel('Frequency', fontsize=10)
        axes[i].grid(True, linestyle=':', linewidth=0.5)
        axes[i].legend(fontsize=9)

    # Hide the last subplot (empty)
    axes[5].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, format='pdf')
    plt.show()

    lin_space = np.linspace(-0.05, 0.05, num=1000)
    true_pdf = MJD_density(lin_space, true_params, dt)
    estimated_pdf = MJD_density(lin_space, mean_estimates, dt)

    plt.figure(figsize=(8, 6))
    plt.plot(lin_space, true_pdf, label="True MJD Density", linewidth=2, color = 'grey')
    plt.plot(lin_space, estimated_pdf, label="Estimated Mean MJD Density", linestyle='--', linewidth=2, color = 'darkblue')
    plt.title("Comparison of True and Estimated Mean MJD PDF", fontsize=12)
    plt.xlabel("Log Returns", fontsize=10)
    plt.ylabel("Density", fontsize=10)
    plt.legend()
    plt.grid(True, linestyle=':', linewidth=0.5)
    plt.show()


# Seed for replication
np.random.seed(1)

# Simulation parameters
dt = 1/252
T = 20

# True parameters
mu_true   = 0.05    # drift
sig_true  = 0.2    # volatility
lam_true  = 20      # jump intensity
muJ_true  = -0.03  # jump mean
sigJ_true = 0.05    # jump volatility
true_params = (mu_true, sig_true, lam_true, muJ_true, sigJ_true)

# Jump Threshold and Penalty
epsilon = 0.03

# Optimization parameters
n_paths = 100
bounds = [(-1, 1), (1e-6, 1), (1e-6, 100), (-1, 1), (1e-6, 1)]

# Parameter storage
estimated_init_params = np.zeros((n_paths, len(true_params)))
estimated_params = np.zeros((n_paths, len(true_params)))

# Loop over simulations
for i in range(n_paths):
    # Simulate one path of log returns
    log_returns = MJD_sim(true_params, T, dt)

    init_params = MJD_init_params(log_returns, dt, epsilon)

    # Estimate parameters via MLE using an initial guess
    result = minimize(
        Loglikelihood,
        x0=init_params,
        # bounds=bounds,
        method="Nelder-Mead",
        args=(log_returns, dt),
        bounds=bounds,
        options={
        'xatol': 1e-5,   # Absolute tolerance between successive iterations in lign with Karlis
        'fatol': 1e-8,   # Optional: function value convergence
        'maxiter': 1000,
        'disp': False
    }
    )

    # Save estimated parameters in array
    estimated_init_params[i, :] = init_params
    estimated_params[i, :] = result.x
    
    print(f"Path {i+1}: Success: {result.success}")

# Compute Mean and RMSE for estimated parameters
mean_estimates = np.mean(estimated_params, axis=0)
rmse_estimates = np.sqrt(np.mean((estimated_params - true_params) ** 2, axis=0))
std_estimates = np.std(estimated_params, axis=0)

# Print statistics
print("\nTrue parameters:", true_params)
print("Mean estimated parameters:", mean_estimates)
print("Std of estimated parameters:", std_estimates)
print("RMSE of estimated parameters:", rmse_estimates)

# Compute Mean and RMSE for init estimated parameters
mean_init_estimates = np.mean(estimated_init_params, axis=0)
rmse_init_estimates = np.sqrt(np.mean((estimated_init_params - true_params) ** 2, axis=0))

# Print statistics
print("Mean estimated init parameters:", mean_init_estimates)
print("RMSE of estimated init parameters:", rmse_init_estimates)

# Plot results
plot_parameter_distributions(estimated_params, true_params)
