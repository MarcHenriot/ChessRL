import numpy as np
from matplotlib import pyplot as plt

def show_board(env):
    env.render()

def plot_for_epochs(num_epochs, tab, title, xlabel, ylabel):
    epochs = np.linspace(0, num_epochs, num = num_epochs)
    plt.plot(epochs, tab, 'r')
    plt.tight_layout()
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.show()