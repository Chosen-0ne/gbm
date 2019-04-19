import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time

def Brownian(seed, N):
    np.random.seed(seed)
    dt = 1. / N  # time step
    b = np.random.normal(0, 1, int(N)) * np.sqrt(dt)  # Brownian increments
    W = np.cumsum(b)  # Brownian path
    return W, b

# GBM Exact Solution

# Parameters
#
# So:     initial stock price
# mu:     returns (drift coefficient)
# sigma:  volatility (diffusion coefficient)
# W:      brownian motion
# T:      time period
# N:      number of increments


def GBM(s0, mu, sigma, W, N, t):
    S = []
    S.append(s0)
    for i in range(1, int(N+1)):
        drift = (mu - 0.5 * sigma**2) * t[i]
        diffusion = sigma * W[i-1]
        S_temp = s0*np.exp(drift + diffusion)
        S.append(S_temp)
    return S


def make_GBM_paths(paths=10, N=500, mu_range=(-0.5, 0.5), sigma_range=(0.1, 0.5), plot=False, save=True):
    start = time.time()
    mu = np.random.uniform(mu_range[0], mu_range[1], paths)
    sigma = np.random.uniform(sigma_range[0], sigma_range[1], paths)

    s0 = 1
    params = pd.DataFrame(columns=['mu', 'sigma', 'seed'])
    GBM_path = pd.DataFrame()
    for mu, sigma in zip(mu, sigma):
        seed = np.random.randint(10000)

        mu = mu
        sigma = sigma
        param_dict_add = {'mu': mu, 'sigma': sigma, 'seed': int(seed)}
        params = params.append(param_dict_add, ignore_index=True)

        w = Brownian(seed, N)[0]
        t = np.linspace(0, 1, N + 1)
        soln = GBM(s0, mu, sigma, w, N, t)   # Exact solution
        GBM_path = GBM_path.append(pd.DataFrame([soln]), ignore_index=True)

        if plot:
            plt.plot(t, soln, linewidth=0.5, label=str(np.round(mu, 2)) + ":" + str(np.round(sigma, 2)))
            plt.ylabel('Stock Price, $')
            plt.title('Geometric Brownian Motion')

    if save:
        params.to_csv('params.csv', sep='\t', index=False)
        GBM_path.to_csv('GBM_paths.csv', sep='\t', index=False, header=False)

    if plot and paths <= 100:
        plt.show()
    elif not plot and paths > 100:
        print("More than 100 GBM paths generated, skipping plotting")
    end = time.time()
    print("Run in {0}s".format(np.round(end - start, 2)))


# make_GBM_paths(N=5000, paths=1, plot=True, mu_range=(0.5, 0.5), sigma_range=(0.4, 0.4))
