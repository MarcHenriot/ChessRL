from matplotlib import pyplot as plt
import os

def show_board(env):
    env.render()

def plot_for_epochs(tab, title, xlabel, ylabel, folder_path=None):
    epochs = [i+1 for i in range(len(tab))]
    plt.plot(epochs, tab, 'r')
    plt.tight_layout()
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    if folder_path: plt.savefig(os.path.join(folder_path, f'{title}.png'))
    plt.show()