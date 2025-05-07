# MJD-NIG-MLE
This code uses MLE to estimate parameters of the Merton Jump Diffusion Model and Normal Inverse Gaussian Model using Swiss market data. A simulation study is included as well.

MJD = Merton Jump Diffusion
NIG = Normal Inverse Gaussian

"Sim Plots" scripts are here to simulate and plot paths of the models, and also to plot the parameter sensitivity of their densities.

"MLE Sim Study" scripts simulate n paths using true and known parameters set by the user and estimate parameters for each path using MLE. Plots and prints of mean and standard deviation show the accuracy of our method.

"MLE Estimation P Bootstrap" scripts estimate the parameters from the "/data/data.csv" file in the Physical measure and reiterate n times using a bootstrap method.

"result summaries/Model Estimation results.xlsx" contains all intermediate results summarized.
