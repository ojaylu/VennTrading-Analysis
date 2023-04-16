import numpy as np
import pandas as pd
import pandas_ta as ta
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math
from implement_stoch_macd_strategy_optimization import *
from rsi import rsi_buy_sell
from obv import *
from ema import *
from flagrules import *
from itertools import product
from datacollection import datacollection
from calc import calc


def bruteforce(rsilb, rsiub):

    surplus = 0
    highestsurplus = -100000
    MACDstrategy = 0
    RSIstrategy = 0
    OBVstrategy = 0
    EMAstrategy = 0
    data = datacollection()
    data = calc(data)
    data = pd.DataFrame(data)
    macdlb = 45
    macdub = 55


    #print("Strategy %s out of %s..." % (i+1, spl))
    RSIflag = data.ta.rsi(close='close', length=14, append=True, signal_indicators=True, xa=rsilb, xb=rsiub)
    macd_buy_price, macd_sell_price, stoch_macd_signal = implement_stoch_macd_strategy_optimization(data['close'], data['%k'], data['%d'], data['macd'], data['macd_signal'], macdlb, macdub)
    rsi_buy_price, rsi_sell_price = rsi_buy_sell(data, RSIflag.columns[1], RSIflag.columns[2])
    obv_buy_price, obv_sell_price = OBV_buy_sell(OBVcalculation(data), 'OBV', 'OBV_EMA')
    data1, signal, ema_buy_price, ema_sell_price = calc_signal(data, 10, 30)
    newMACDPriceBuy = [item for item in macd_buy_price if not(math.isnan(item)) == True]
    newMACDPriceSell = [item for item in macd_sell_price if not(math.isnan(item)) == True]
    newRSIPriceBuy = [item for item in rsi_buy_price if not(math.isnan(item)) == True]
    newRSIPriceSell = [item for item in rsi_sell_price if not(math.isnan(item)) == True]
    newOBVPriceBuy = [item for item in obv_buy_price if not(math.isnan(item)) == True]
    newOBVPriceSell = [item for item in obv_sell_price if not(math.isnan(item)) == True]
    newEMAPriceBuy = [item for item in ema_buy_price if not(math.isnan(item)) == True]
    newEMAPriceSell = [item for item in ema_sell_price if not(math.isnan(item)) == True]
    surplus = sum(newMACDPriceSell) + sum(newRSIPriceSell) + sum(newOBVPriceSell) + sum(newEMAPriceBuy)
    - sum(newMACDPriceBuy) - sum(newRSIPriceBuy) - sum(newOBVPriceBuy) - sum(newEMAPriceSell)

    #print("This strategy help you earn",surplus)

    # MACDstrategy, RSIstrategy, OBVstrategy, EMAstrategy, combineflag, data, newsurplus = flagrules(
    #     data, MACDstrategy, RSIstrategy, OBVstrategy, EMAstrategy)

    return -surplus

# Define the PSO algorithm
def pso(cost_func, dim=2, num_particles=30, max_iter=10, w=0.5, c1=1, c2=2):
    # Initialize particles and velocities
    particles = np.random.uniform(0, 100, (num_particles, dim))
    velocities = np.zeros((num_particles, dim))

    print('particles is', particles)

    # Initialize the best positions and fitness values
    best_positions = np.copy(particles)
    best_fitness = np.array([cost_func(a,b) for a,b in particles])
    swarm_best_position = best_positions[np.argmin(best_fitness)]
    swarm_best_fitness = np.min(best_fitness)

    # Iterate through the specified number of iterations, updating the velocity and position of each particle at each iteration
    for i in range(max_iter):
        # Update velocities
        r1 = np.random.uniform(0, 1, (num_particles, dim))
        r2 = np.random.uniform(0, 1, (num_particles, dim))
        velocities = w * velocities + c1 * r1 * (best_positions - particles) + c2 * r2 * (swarm_best_position - particles)

        # Update positions
        particles += velocities

        # Evaluate fitness of each particle
        fitness_values = np.array([cost_func(a,b) for a,b in particles])

        # Update best positions and fitness values
        improved_indices = np.where(fitness_values < best_fitness)
        best_positions[improved_indices] = particles[improved_indices]
        best_fitness[improved_indices] = fitness_values[improved_indices]
        if np.min(fitness_values) < swarm_best_fitness:
            swarm_best_position = particles[np.argmin(fitness_values)]
            swarm_best_fitness = np.min(fitness_values)

    # Return the best solution found by the PSO algorithm
    return swarm_best_position, swarm_best_fitness

# Define the dimensions of the problem
dim = 2

# Run the PSO algorithm on the Rastrigin function
solution, fitness = pso(bruteforce, dim=dim)

# Print the solution and fitness value
print('Solution:', solution)
print('Fitness:', fitness)

# Create a meshgrid for visualization
x = np.linspace(0, 100, 100)
y = np.linspace(0, 100, 100)
X, Y = np.meshgrid(x, y)
Z = bruteforce(X, Y)

# Create a 3D plot of the Rastrigin function
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z, cmap='viridis')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')

# Plot the solution found by the PSO algorithm
ax.scatter(solution[0], solution[1], fitness, color='red')
plt.show()