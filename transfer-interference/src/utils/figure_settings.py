import matplotlib as mpl

med_fontsize = 15
schedule_colours = ['#E69F00', '#56B4E9', '#D9544D']
condition_order = ['same', 'near', 'far']
task_colours = ['#b08cca','#aad39b']
cm_conv = 1 / 2.54  

# Matplotlib style settings
mpl.rcParams['axes.linewidth'] = 0.5
mpl.rcParams['xtick.major.width'] = 0.5
mpl.rcParams['ytick.major.width'] = 0.5
mpl.rcParams['xtick.minor.width'] = 0.25
mpl.rcParams['ytick.minor.width'] = 0.25
# Use DejaVu Sans (common on Linux) or fall back to sans-serif
# If you need Arial specifically, install: sudo apt-get install ttf-mscorefonts-installer
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica', 'sans-serif']
mpl.rcParams['font.size'] = 8
mpl.rcParams['axes.titlesize'] = 8
mpl.rcParams['axes.labelsize'] = 8
mpl.rcParams['xtick.labelsize'] = 8
mpl.rcParams['ytick.labelsize'] = 8
mpl.rcParams['legend.fontsize'] = 8
mpl.rcParams['figure.dpi'] = 300
mpl.rcParams['legend.frameon'] = False
