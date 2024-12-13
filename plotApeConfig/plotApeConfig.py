# plot_config.py
import matplotlib.pyplot as plt
from cycler import cycler

blueAPE='#000077'
grayConcrete='#e4e4e4'

def set_default_plot_params():
    plt.rcParams['font.family'] = 'arial'
    plt.rcParams['font.size'] = 10
    plt.rcParams['font.stretch'] = 'condensed'
    plt.rcParams['grid.linestyle'] = '--'
    plt.rcParams['grid.linewidth'] = 0.5
    plt.rcParams['axes.grid'] = True
    # Set the color cycle globally
    plt.rcParams['axes.prop_cycle'] = cycler(color=color_palette)
    
color_palette = [
    blueAPE,
    'k',
    'blue',
    '#999999',  # Dark Gray
    grayConcrete,
    '#4daf4a',  # Dark Green
    '#984ea3',  # Dark Purple
    '#ff7f00',  # Dark Orange
    '#e41a1c',  # Dark Red
    '#ffff33',  # Dark Yellow
    '#a65628',  # Dark Brown
    '#377eb8',  # Dark Blue
    '#f781bf',  # Dark Pink
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
    '#b2df8a',   # Dark Mint Green
    '#2ecc71',  # Emerald Green
    '#3498db',  # Bright Blue
    '#9b59b6',  # Amethyst Purple
    '#f1c40f',  # Sunflower Yellow
    '#e67e22',  # Carrot Orange
    '#e74c3c',  # Alizarin Red
    '#1abc9c',  # Turquoise
    '#16a085',  # Green Sea
    '#27ae60',  # Nephritis Green
    '#2980b9',  # Belize Blue
    '#8e44ad',  # Wisteria Purple
    '#f39c12',  # Orange
    '#d35400',  # Pumpkin
    '#c0392b',  # Pomegranate
    '#bdc3c7',  # Silver
    '#7f8c8d',  # Asbestos Gray
    '#2c3e50',  # Midnight Blue
    '#ea4c88',  # Pink Rose
    '#8e8e93',  # Gray
    '#34495e',  # Wet Asphalt
    '#5856d6',  # Electric Purple
    '#ff2d55',  # Red Pink
    '#5ac8fa',  # Light Blue
    '#007aff',  # Azure Blue
    '#4cd964',  # Spring Green
    '#ff9500',  # Orange Yellow
    '#ff3b30',  # Red Orange
    '#8a8a8f',  # Cool Gray
    '#ceced2',  # Light Gray
    '#c7c7cc',  # System Gray
    '#ff4981',  # Bright Pink
    '#4a90e2',  # Sky Blue
    '#f5a623',  # Golden Yellow
    '#50e3c2',  # Aqua Green
    '#b8e986',  # Light Green
    '#bd10e0',  # Bright Purple
    '#9013fe',  # Deep Purple
    '#4a4a4a',  # Dark Gray 2
    '#9b9b9b',  # Medium Gray
    '#d0021b',  # Bright Red
    '#f8e71c',  # Bright Yellow
    '#7ed321',  # Lime Green
    '#417505',  # Forest Green
    '#bd10e0',  # Magenta Purple
    '#9013fe',  # Electric Violet
    '#8b572a',  # Brown
    '#417505',  # Deep Green
    '#9013fe',  # Rich Purple
    '#50e3c2',  # Turquoise Blue
    '#4a90e2',  # Ocean Blue
    '#ff7a00',  # Bright Orange
    '#ff0000',  # Pure Red
    '#00ff00',  # Pure Green
    '#0000ff',  # Pure Blue
    '#ff00ff',  # Pure Magenta
    '#00ffff',  # Pure Cyan
    '#ffff00',  # Pure Yellow
    '#800000',  # Maroon
    '#008000',  # Dark Green 2
    '#000080',  # Navy Blue
    '#800080',  # Purple
    '#008080',  # Teal
    '#808000',  # Olive
    '#ff1493',  # Deep Pink
    '#00bfff',  # Deep Sky Blue
    '#ff4500',  # Orange Red
    '#da70d6',  # Orchid
    '#32cd32',  # Lime Green 2
    '#6a5acd',  # Slate Blue
    '#40e0d0',  # Turquoise 2
    '#ee82ee',  # Violet
    '#ffa07a',  # Light Salmon
    '#87ceeb',  # Sky Blue 2
    '#98fb98',  # Pale Green
    '#dda0dd',  # Plum
    '#f0e68c',  # Khaki
    '#ff69b4',  # Hot Pink
    '#cd853f',  # Peru
    '#ffdead',  # Navajo White
    '#b8860b',  # Dark Goldenrod
    '#4b0082',  # Indigo
    '#556b2f',  # Dark Olive Green
    '#8b0000',  # Dark Red 2
    '#006400',  # Dark Green 3
    '#8b008b',  # Dark Magenta 2
    '#191970',  # Midnight Blue 2
    '#bdb76b',  # Dark Khaki
    '#8b4513',  # Saddle Brown
    '#e9967a',  # Dark Salmon
    '#9400d3',  # Dark Violet 2
    '#ff8c00',  # Dark Orange 2
    '#00ced1',  # Dark Turquoise
    '#9370db',  # Medium Purple
    '#3cb371',  # Medium Sea Green
    '#20b2aa',  # Light Sea Green
    '#778899',  # Light Slate Gray
    '#b0c4de',  # Light Steel Blue
    '#ffb6c1',  # Light Pink
    '#00fa9a',  # Medium Spring Green
    '#48d1cc',  # Medium Turquoise
    '#c71585',  # Medium Violet Red
    '#191970'   # Midnight Blue 3
]