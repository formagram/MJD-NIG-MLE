import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.special import k1
from scipy.optimize import minimize

def NIG_init_params(data: np.ndarray, dt: float) -> tuple:

    # Compute sample moments
    mu_bar   = np.mean(data)
    s_bar    = np.std(data)
    var_bar  = np.mean(np.power(data - mu_bar, 2))
    skew_bar = np.mean(np.power(data - mu_bar, 3))
    kurt_bar = np.mean(np.power(data - mu_bar, 4))

    # sample skewness and kurtosis
    gamma_1 = skew_bar / var_bar**(3 / 2)
    gamma_2 = kurt_bar / var_bar**2 - 3

    # Condition check
    moment_check = 3 * gamma_2 - 5 * gamma_1**2

    if moment_check <= 0:
        # Moment condition fails → return NaNs or fallback
        print("Moment condition invalid: skipping this sample.")
        return (np.nan, np.nan, np.nan, np.nan)

    # Compute as shown from papers
    gamma_hat = 3 / (s_bar * np.sqrt(3 * gamma_2 - 5 * gamma_1**2))
    beta_hat  = (gamma_1 * s_bar * gamma_hat**2) / 3
    delta_hat = (s_bar**2 * gamma_hat**3) / (beta_hat**2 + gamma_hat**2)
    mu_hat    = mu_bar - beta_hat * delta_hat / gamma_hat
    alpha_hat = np.sqrt(gamma_hat**2 + beta_hat**2)

    # Scale with dt
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

# Load your data, and subset if you wish
data = pd.read_csv(r"data/data.csv", parse_dates=True, header=0, skip_blank_lines=True).dropna().set_index("Date")
data.index = pd.to_datetime(data.index)
start_date = '2000-01-03'
end_date = '2025-01-03' 
data = data.loc[start_date:end_date]

# Args
dt = 1/252 # time increment
bounds = [(-np.inf, np.inf), (1e-6, None), (-300, 300), (1e-6, None)]
n_resamples = 1000 # n bootstrapp samples
k = 4 # number of parameters

# Iterate for each symbol
for i in range(0,len(data.columns)):

    # Result matrices
    bootstrapped_res = np.zeros((n_resamples, k))
    bootstrapped_init_res = np.zeros((n_resamples, k))

    # Extract stock returns and name
    log_returns = data.iloc[:,i].values
    symbol = data.iloc[:,i].name

    print("")
    print(f"Symbol estimated: {symbol}")

    # Iterate through bootstrap samples
    for j in range(n_resamples):

        print(j)
        
        if j == 0:
            resampled_returns = log_returns  # Use original data
        else:
            resampled_returns = np.random.choice(log_returns, size=len(log_returns), replace=True) # Uniform resampling

        # Compute init params for each resample
        init_params= NIG_init_params(resampled_returns, dt)

        # Minimize
        result = minimize(
            Loglikelihood,
            x0=init_params, 
            bounds=bounds,
            method="Nelder-Mead",
            args=(resampled_returns, dt, ),
            options={
            'xatol': 1e-5,   # Absolute tolerance between successive iterations in lign with Karlis
            'fatol': 1e-8,   # Optional: function value convergence
            'maxiter': 10000,
            'disp': False
        }
        )

        # Uncomment for results of AIC BIC
        # n = len(log_returns)
        # neg_log_likelihood = result.fun

        # aic = 2 * k + 2 * neg_log_likelihood
        # bic = k * np.log(n) + 2 * neg_log_likelihood

        # print(f"AIC: {aic:.2f}, BIC: {bic:.2f}")

        if result.success == False:
            print("Success: ", result.success)
            print("Success: ", result)

        bootstrapped_res[j, :] = result.x
        bootstrapped_init_res[j, :] = init_params

    # Clean from nan's to prevent errors (Should not be the case though)
    bootstrapped_res = bootstrapped_res[~np.isnan(bootstrapped_res).any(axis=1)]
    bootstrapped_init_res = bootstrapped_init_res[~np.isnan(bootstrapped_init_res).any(axis=1)]

    # Compute stats from bootstrapped results
    estimation_mean = np.mean(bootstrapped_res, axis=0)
    estimation_deviation = np.std(bootstrapped_res, axis=0)
    bootstrap_deviation = np.sqrt(np.mean((bootstrapped_res - bootstrapped_res[0, :]) ** 2, axis=0))
    t_stat = np.abs(estimation_mean) / bootstrap_deviation

    # Compute stats from bootstrapped init results
    estimation_init_mean = np.mean(bootstrapped_init_res, axis=0)
    estimation_init_deviation = np.std(bootstrapped_init_res, axis=0)
    bootstrap_init_deviation = np.sqrt(np.mean((bootstrapped_init_res - bootstrapped_init_res[0, :]) ** 2, axis=0))
    t_stat_init = np.abs(estimation_init_mean) / bootstrap_init_deviation

    # Print results
    print("Estimated Parameters:")
    print(estimation_mean)
    print(estimation_deviation)
    print(bootstrap_deviation)
    print(t_stat)

    print("")

    print("Initial Prameter:")
    print(estimation_init_mean)
    print(estimation_init_deviation)
    print(bootstrap_init_deviation)
    print(t_stat_init)

    # Plot comparison between true PDF and mean estimated PDF
    lin_space = np.linspace(min(log_returns), max(log_returns), 1000)
    estimated_pdf = NIG_density(lin_space, estimation_mean, dt)

    plt.figure(figsize=(8, 6))
    plt.hist(log_returns, bins=100, alpha=0.7, color='grey', edgecolor='black', density=True)
    plt.plot(lin_space, estimated_pdf, label="Density based on bootstrap mean", linestyle='-', linewidth=2, color = 'darkblue')
    plt.title(f"Estimated NIG Density for Symbol: {symbol}", fontsize=12)
    plt.xlabel("Log Returns", fontsize=10)
    plt.ylabel("Density", fontsize=10)
    plt.legend()
    plt.grid(True, linestyle=':', linewidth=0.5)
    plt.show()


    
