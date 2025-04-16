import matplotlib.pyplot as plt
import numpy as np

plt.figure()
for y in range(1, 5):
    plt.bar(x=[0, 1], height=[1, y], color='green')
    plt.draw()
    plt.pause(2)
