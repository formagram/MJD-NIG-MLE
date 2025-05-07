import numpy as np
import matplotlib.pyplot as plt
from scipy.special import k1
from scipy.optimize import minimize

def NIG_sim(params: tuple, T: int, dt: float) -> np.ndarray:
   
   mu, alpha, beta, delta = params
   steps = int(T * (1/dt))

   gamma = np.sqrt(alpha**2 - beta**2)

   IG = np.random.wald(delta * dt / gamma, (delta * dt) ** 2, steps)
   dW = np.random.normal(0, 1, steps)

   log_returns = mu * dt + beta * IG + dW * np.sqrt(IG)

   return log_returns

def NIG_init_params(data: np.ndarray, dt: float) -> tuple:

    mu_bar   = np.mean(data)
    s_bar    = np.std(data)
    var_bar  = np.mean(np.power(data - mu_bar, 2))
    skew_bar = np.mean(np.power(data - mu_bar, 3))
    kurt_bar = np.mean(np.power(data - mu_bar, 4))

    gamma_1 = skew_bar / var_bar**(3 / 2)
    gamma_2 = kurt_bar / var_bar**2 - 3

    moment_check = 3 * gamma_2 - 5 * gamma_1**2
    if moment_check <= 0:
        # Moment condition fails → return NaNs or fallback
        print("Moment condition invalid: skipping this sample.")
        return (np.nan, np.nan, np.nan, np.nan)

    gamma_hat = 3 / (s_bar * np.sqrt(3 * gamma_2 - 5 * gamma_1**2))
    beta_hat  = (gamma_1 * s_bar * gamma_hat**2) / 3
    delta_hat = (s_bar**2 * gamma_hat**3) / (beta_hat**2 + gamma_hat**2)
    mu_hat    = mu_bar - beta_hat * delta_hat / gamma_hat
    alpha_hat = np.sqrt(gamma_hat**2 + beta_hat**2)

    init_params = (mu_hat * dt**-1, alpha_hat, beta_hat, delta_hat * dt**-1)

    return init_params

def NIG_density(x: np.ndarray, params: tuple, dt: float) -> np.ndarray:

    # Unpack parameters
    mu, alpha, beta, delta = params

    # Adjust for time step
    mu = mu * dt
    delta = delta * dt

    # Compute args
    gamma = np.sqrt(alpha**2 - beta**2)
    arg = np.sqrt(delta**2 + np.power(x - mu, 2))

    # Compute density components
    numerator = alpha * delta * k1(alpha * arg)
    denominator = np.pi * arg
    exponent = np.exp(delta * gamma + beta * (x - mu))

    # Assemble to density
    density = (numerator / denominator) * exponent
    return np.maximum(density, 1e-30)

def Loglikelihood(params: np.ndarray, data: np.ndarray, dt: float) -> float:

    # Unpack parameters
    mu, alpha, beta, delta = params

    # Check model boundaries
    if alpha <= 0 or delta <= 0 or abs(beta) >= alpha:
        return np.inf
    
    # Compute density vector and clip too low values
    density = NIG_density(data, (mu, alpha, beta, delta), dt)
    density = np.clip(density, 1e-10, None)
    

    # Compute negative log-likelihood
    log_likelihood = -np.sum(np.log(density))

    return log_likelihood

def plot_parameter_distributions(estimated_params: np.ndarray, true_params: tuple, save_path="nig_param_distributions.pdf"): 

    param_names = ['μ', 'α', 'β', 'δ']  # Greek symbols for elegance
    true_values = list(true_params)

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    axes = axes.flatten()

    for i in range(4):
        axes[i].hist(estimated_params[:, i], bins=20, alpha=0.7, color='grey', edgecolor='black')
        axes[i].axvline(true_values[i], color='black', linestyle='--', linewidth=1.5, label=f'True {param_names[i]} = {true_values[i]:.4f}')
        mean_est = np.mean(estimated_params[:, i])
        axes[i].axvline(mean_est, color='darkblue', linestyle='-', linewidth=1.5, label=f'Mean = {mean_est:.4f}')
        axes[i].set_title(f'Estimated {param_names[i]} Distribution', fontsize=12)
        axes[i].set_xlabel(f'{param_names[i]} Value', fontsize=10)
        axes[i].set_ylabel('Frequency', fontsize=10)
        axes[i].grid(True, linestyle=':', linewidth=0.5)
        axes[i].legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, format='pdf')
    plt.close()

    # Plot comparison between true PDF and mean estimated PDF
    lin_space = np.linspace(-0.05, 0.05, num=1000)
    true_pdf = NIG_density(lin_space, true_params, dt)
    estimated_pdf = NIG_density(lin_space, mean_estimates, dt)

    plt.figure(figsize=(8, 6))
    plt.plot(lin_space, true_pdf, label="True NIG Density", linewidth=2, color = 'grey')
    plt.plot(lin_space, estimated_pdf, label="Estimated Mean NIG Density", linestyle='--', linewidth=2, color = 'darkblue')
    plt.title("Comparison of True and Estimated Mean NIG PDF", fontsize=12)
    plt.xlabel("Log Returns", fontsize=10)
    plt.ylabel("Density", fontsize=10)
    plt.legend()
    plt.grid(True, linestyle=':', linewidth=0.5)
    plt.show()



def NIG_implied_moments(params: tuple, dt: float) -> tuple:
    mu, alpha, beta, delta = params

    gamma = np.sqrt(alpha**2 - beta**2)

    imp_mean = mu + delta * beta / gamma
    imp_var  = delta * alpha**2 / gamma**3
    imp_skew = 3 * beta / (alpha * np.sqrt(gamma * delta))
    imp_kurt = 3 * (1 + 4 * beta**2 / alpha**2)/(delta * gamma)

    imp_mean_annu = imp_mean * dt**(-1)
    imp_std_annu = np.sqrt(imp_var  * dt**(-1))

    return (imp_mean_annu, imp_std_annu, imp_skew, imp_kurt)

# Seed for replication
np.random.seed(1)

# Simulation parameters
dt = 1/252
T = 20

# True parameters
mu      = 0.05
alpha   = 30
beta    = -10
delta   = (0.3)
true_params = (mu, alpha, beta, delta)

# Optimization parameters
n_paths = 100
bounds = [(-1, 1), (1e-6, np.inf), (-300, 300), (1e-6, 1)]

# Parameter storage
estimated_init_params = np.zeros((n_paths, len(true_params)))
estimated_params = np.zeros((n_paths, len(true_params)))

#
for i in range(n_paths):
    log_returns = NIG_sim(true_params, T, dt)

    init_params = NIG_init_params(log_returns, dt)

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

estimated_params = estimated_params[~np.isnan(estimated_params).any(axis=1)]

# Compute Mean and RMSE for estimated parameters
mean_estimates = np.mean(estimated_params, axis=0)
rmse_estimates = np.sqrt(np.mean((estimated_params - true_params) ** 2, axis=0))
std_estimates = np.std(estimated_params, axis=0)

# Print statistics
print("\nTrue parameters:", true_params)
print("Mean estimated parameters:", mean_estimates)
print("Std of estimated parameters:", std_estimates)
print("RMSE of estimated parameters:", rmse_estimates)

# Call the plotting function
plot_parameter_distributions(estimated_params, true_params)

