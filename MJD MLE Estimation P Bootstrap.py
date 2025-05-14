import numpy as np
import pandas as pd
import scipy.stats as ss
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from math import factorial


def MJD_init_params(data: np.ndarray, dt: float, epsilon: float) -> tuple:

    # Separate data accoridng to threshold
    returns_no_jump = data[np.abs(data) < epsilon]
    returns_jump    = data[np.abs(data) >= epsilon]

    # Estimate jump intensity
    lambda_hat = len(returns_jump)/(len(data) * dt)

    # Estimate parameters when no jump
    mu_hat  = (2*np.mean(returns_no_jump) + np.var(returns_no_jump)*dt)/(2*dt)
    sig_hat = np.sqrt(np.var(returns_no_jump)/dt)
    
    # Estimate jump parameters
    muJ_hat = np.mean(returns_jump) - (mu_hat - (sig_hat**2)/2)*dt
    sigJ_hat = np.sqrt(max(0, np.var(returns_jump) - (sig_hat**2)*dt))

    return (mu_hat, sig_hat, lambda_hat, muJ_hat, sigJ_hat)

def MJD_density(x: np.ndarray, params: tuple, dt: float) -> np.ndarray:
    
    # Unpack parameters
    mu, sig, lam, muJ, sigJ = params

    # Init density
    density = np.zeros_like(x)

    # Jump compensation term
    k = np.exp(muJ + (sigJ**2) / 2) - 1 

    # Sum the contributions for 0 to 10 jumps for the density function below
    for n in range(10):
        mu_den = (mu - (sig**2) / 2 - lam * k) * dt + n * muJ
        var_den = (sig**2) * dt + (sigJ**2) * n
        var_den = np.maximum(var_den, 1e-30)  # avoid zero variance
        norm_den = (1 / np.sqrt(2 * np.pi * var_den)) * np.exp(-((x - mu_den)**2) / (2 * var_den))
        poisson_den = np.exp(-lam * dt) * ((lam * dt)**n) / factorial(n) # Poisson
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


# Load your data, and subset if you wish
data = pd.read_csv(r"data/data.csv", parse_dates=True, header=0, skip_blank_lines=True).dropna().set_index("Date")
data.index = pd.to_datetime(data.index)
start_date = '2000-01-03'
end_date = '2025-01-03' 
data = data.loc[start_date:end_date]

# Args
dt = 1/252         # time increment
epsilon = 0.03     # Jump threshold
bounds = [(-1, 1), (1e-6, 1), (1e-6, 100), (-1, 1), (1e-6, 1)]
n_resamples = 1000 # number of bootstrap smaples
k = 5              # k parameters estimater

# Iterate for each symbol
for i in range(0,len(data.columns)):

    # Result matrices
    bootstrapped_res = np.zeros((n_resamples, k))
    bootstrapped_init_res = np.zeros((n_resamples, k))

    # Extract return values and name
    log_returns = data.iloc[:,i].values
    symbol = data.iloc[:,i].name

    print("")
    print(f"Symbol estimated: {symbol}")
    
    # Iterate through bootstrap samples
    for j in range(n_resamples):
        
        if j == 0:
            resampled_returns = log_returns  # Use original data
        else:
            resampled_returns = np.random.choice(log_returns, size=len(log_returns), replace=True) # Uniform resampling

        # Compute init params for each resample
        init_params= MJD_init_params(resampled_returns, dt, epsilon)

        # Minimize
        result = minimize(
            Loglikelihood,
            x0=init_params,
            bounds=bounds,
            method="Nelder-Mead",
            args=(resampled_returns, dt, ),
            options={
            'xatol': 1e-5,   # Absolute tolerance between successive iterations in line with Karlis
            'fatol': 1e-8,   
            'maxiter': 10000,
            'disp': False
        }
        )

        # # Uncomment below for results of AIC BIC, I ran it with n_resamples = 1 to have only the values for original estimate
        # n = len(log_returns)
        # neg_log_likelihood = result.fun

        # aic = 2 * k + 2 * neg_log_likelihood
        # bic = k * np.log(n) + 2 * neg_log_likelihood

        # print(f"AIC: {aic:.2f}, BIC: {bic:.2f}")

        if result.success == False:
            print("Success: ", result.success)
            print(result)

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
    estimated_pdf = MJD_density(lin_space, estimation_mean, dt)

    plt.figure(figsize=(8, 6))
    plt.hist(log_returns, bins=100, alpha=0.7, color='grey', edgecolor='black', density=True)
    plt.plot(lin_space, estimated_pdf, label="Density based on bootstrap mean", linestyle='-', linewidth=2, color = 'darkblue')
    plt.title(f"Estimated MJD Density for Symbol: {symbol}", fontsize=12)
    plt.xlabel("Log Returns", fontsize=10)
    plt.ylabel("Density", fontsize=10)
    plt.legend()
    plt.grid(True, linestyle=':', linewidth=0.5)
    plt.show()

