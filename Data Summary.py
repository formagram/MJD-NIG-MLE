import numpy as np
import pandas as pd
import scipy.stats as ss

# Load your data, and subset if you wish
data = pd.read_csv(r"data/data.csv", parse_dates=True, header=0, skip_blank_lines=True).dropna().set_index("Date")
data.index = pd.to_datetime(data.index)
start_date = '2000-01-03' # 2000-01-03, 2012-01-05
end_date = '2025-01-03'  # 2012-01-05, 2025-01-03
data = data.loc[start_date:end_date]

# Iterate for each symbol
for i in range(0,len(data.columns)):
    log_returns = data.iloc[:,i].values

    symbol = data.iloc[:,i].name

    mean = np.mean(log_returns)
    std = np.std(log_returns, ddof=1)
    skew = ss.skew(log_returns, bias=False)
    kurt = ss.kurtosis(log_returns, bias=False)

    print("")
    print(f"Symbol estimated: {symbol}")
    print(f"Mean: {mean*252}")
    print(f"Standard Deviation: {std*np.sqrt(252)}")
    print(f"Skewness: {skew}")
    print(f"Fisher Kurtosis: {kurt}")
    print("")

    ks_statistic, p_value = ss.kstest(log_returns, 'norm', args=(mean, std))

    print(f"KS Statistic: {ks_statistic}")
    print(f"P-Value: {p_value}")

    # Interpretation
    alpha = 0.01
    if p_value < alpha:
        print("Reject the null hypothesis: Data is not normally distributed.")
    else:
        print("Fail to reject the null hypothesis: Data appears to be normally distributed.")

