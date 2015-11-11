import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

def pretty_mpl_defaults():
    # Sources:
    # http://nbviewer.ipython.org/gist/olgabot/5357268
    # http://stackoverflow.com/questions/24808739/modifying-the-built-in-colors-of-matplotlib
    # Font size
    # used 32 for ISI abstract
    # used 18 for slide plots
    # used 24 for 3 plots / page
    mpl.rcParams['font.size'] = 20
    # Same for legend font size
    mpl.rcParams['legend.fontsize'] = mpl.rcParams['axes.labelsize']
    mpl.rcParams['legend.fancybox'] = True
    # Line width
    mpl.rcParams['lines.linewidth'] = 2
    mpl.rcParams['lines.markersize'] = 10
    # Set colormap to a sequential one
    mpl.rcParams['image.cmap'] = 'Blues' # 'Greys'
    # The circles surrounding scatters (also legend boxes...)
    mpl.rcParams['patch.edgecolor'] = 'white'
    mpl.rcParams['patch.linewidth'] = 0.2
    
    newcycle = plt.cm.Set1(np.linspace(0,1,9)).tolist()
    cycle = [x[:3] for x in newcycle][:7]
    # Get the order of the first few colors right: blue, green, red
    # Anything else doesn't matter that much
    tmp = cycle[0]
    cycle.remove(tmp)
    cycle.insert(2,tmp)
    mpl.rcParams['axes.color_cycle'] = cycle

    import matplotlib.colors as colors
    cdict = colors.colorConverter.colors
    cdict['b'] = cycle[0]
    cdict['g'] = cycle[1]
    cdict['r'] = cycle[2]
    cdict['y'] = cycle[5]
    # This match is quite a stretch
    cdict['c'] = cycle[3]
    cdict['m'] = cycle[4]
    # Just for completeness
    cdict['w'] = cdict['w']
    cdict['b'] = cdict['b']
    colors.colorConverter.cache = {}
    
    #Scientific notation
    mpl.rcParams['axes.formatter.limits'] = [-3,3]
