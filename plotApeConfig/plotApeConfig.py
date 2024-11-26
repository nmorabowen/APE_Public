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
    
color_palette = [
    blueAPE,
    'k',
    '#4daf4a',  # Dark Green
    grayConcrete,
    '#984ea3',  # Dark Purple
    '#ff7f00',  # Dark Orange
    '#e41a1c',  # Dark Red
    '#ffff33',  # Dark Yellow
    '#a65628',  # Dark Brown
    '#377eb8',  # Dark Blue
    '#f781bf',  # Dark Pink
    '#999999',  # Dark Gray
    '#dede00',  # Dark Lime
    '#1b9e77',  # Dark Teal
    '#d95f02',  # Dark Burnt Orange
    '#7570b3',  # Dark Violet
    '#e7298a',  # Dark Magenta
    '#66a61e',  # Dark Olive
    '#e6ab02',  # Dark Mustard
    '#a6761d',  # Dark Taupe
    '#666666',  # Dark Charcoal
    '#1f78b4',  # Dark Sky Blue
    '#b2df8a'   # Dark Mint Green
]