# plot_config.py
import matplotlib.pyplot as plt

blueAPE='#000077'
grayConcrete='#e4e4e4'

def set_default_plot_params():
    plt.rcParams['font.family'] = 'arial'
    plt.rcParams['font.size'] = 10
    plt.rcParams['font.stretch'] = 'condensed'
    plt.rcParams['grid.linestyle'] = '--'
    plt.rcParams['grid.linewidth'] = 0.5
    plt.rcParams['axes.grid'] = True