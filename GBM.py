import numpy as np
import matplotlib.pyplot as plt

def Brownian(seed, N):
    np.random.seed(seed)
    dt = 1. / N  # time step
    b = np.random.normal(0., 1., int(N)) * np.sqrt(dt)  # brownian increments
    W = np.cumsum(b)  # brownian path
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


def GBM(So, mu, sigma, W, N):
    S = []
    S.append(So)
    for i in range(1, int(N+1)):
        drift = (mu - 0.5 * sigma**2) * t[i]
        diffusion = sigma * W[i-1]
        S_temp = So*np.exp(drift + diffusion)
        S.append(S_temp)
    return S, t


seed = np.random.randint(10000)
N = 100000
So = 1
mu = 0
sigma = 0.4


W = Brownian(seed, N)[0]
t = np.linspace(0, 1, N + 1)
soln = GBM(So, mu, sigma, W, N)[0]    # Exact solution


plt.plot(t, soln, linewidth=0.75)
plt.ylabel('Stock Price, $')
plt.title('Geometric Brownian Motion')
plt.show()