"""
    Thomas DeWitt
    Functions for plotting data.
"""
import os
from sys import exit
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.patches as patches
from scipy.interpolate import interpn
from scipy.stats import gaussian_kde, linregress, t
import directories
import imageio
import n2str
from sigfig import round as siground

matplotlib.rc('text', usetex=True)
dirs = directories.StringDirectories()

def main():
    
    fig, ax = plt.subplots(1,1)
    ax.plot([],[])
    ax.set_xlim(-3,20)
    ax.set_ylim(-3,10)
    format_log_log(ax)
    savefig('test', True)


# Global vars:
cloud_colors_grey = matplotlib.colors.ListedColormap(["#025373", "#EEEEEE"], name='cloud_colors_grey', N=None)
cloud_colors = matplotlib.colors.ListedColormap(["#025373", "white"], name='cloud_colors', N=None)
anti_cloud_colors = matplotlib.colors.ListedColormap(["#FF8500", "#FF8500"], name='anti_cloud_colors', N=None)
pink_colors = matplotlib.colors.ListedColormap(["#800080", "#800080"], name='pink_colors', N=None)
green_colors = matplotlib.colors.ListedColormap(["#01730B", "#01730B"], name='green_colors', N=None)
cloud_colors.set_bad(color='#8A8A8A')
cont_cloud_colors = matplotlib.colors.LinearSegmentedColormap.from_list('white_viridis', [
    (0, '#025373'),
    (1, '#FFFFFF'),
], N=256)
cont_cloud_colors.set_bad(color='#8A8A8A')
white_viridis = matplotlib.colors.LinearSegmentedColormap.from_list('white_viridis', [
    (0, '#ffffff'),
    (1e-10, '#440053'),
    (0.01, '#404388'),
    (0.03, '#2a788e'),
    (0.1, '#21a784'),
    (0.3, '#78d151'),
    (1, '#fde624'),
], N=256)
log_viridis = matplotlib.colors.LinearSegmentedColormap.from_list('log_viridis', [
    (0, '#440053'),
    (0.001, '#404388'),
    (0.01, '#2a788e'),
    (0.1, '#21a784'),
    (0.3, '#78d151'),
    (1, '#fde624'),
], N=256)

# Plotting tools:
def place_legend_anywhere(fig, ax, text, x_loc, y_loc, marker='o', color='black', fontsize=10):
    """
        Make legend for single series of data, with enlarged marker as a key, and place anywhere on ax.

        Input:
            fig - plt.figure instance
            ax - plt.Axes instance
            text - Text to make into legend: str
            x_loc - Location in data coords of text: float
            y_loc - Location in data coords of text: float
            marker - Marker to put in legend: str
                        Some options (plt Markers):
                            'o'
                            'v'
                            'p'
            color - Color to make marker: str
            fontsize - Passed to ax.text: int
        Output: 
            ax - plt.Axes with text placed

    """

    t = ax.text(x_loc, y_loc, text, fontdict={'fontsize': fontsize})
    
    # Get size of text:
    r = fig.canvas.get_renderer()
    bb = matplotlib.transforms.Bbox(ax.transData.inverted().transform(t.get_window_extent(renderer=r)))
    height = bb.height

    # Place marker:
    marker_x = x_loc - (ax.get_xlim()[1]-ax.get_xlim()[0])/40 # put a little to the left
    marker_y = y_loc + 0.5*height
    ax.scatter([marker_x],[marker_y], color=color,s=4*fontsize, marker=marker)

    return ax

def density_scatter(x ,y,ax, sort = True, bins = 100, log_colors=True, **kwargs )   :
    """
    Creates a scatter plot where color corresponds to point density.
    kwargs passed to ax.scatter
    log_colors: Whether to logarithmically scale the colormap
    From https://stackoverflow.com/a/53865762/3015186
    """
    data , x_e, y_e = np.histogram2d( x, y, bins = bins, density = True )
    z = interpn( ( 0.5*(x_e[1:] + x_e[:-1]) , 0.5*(y_e[1:]+y_e[:-1]) ) , data , np.vstack([x,y]).T , method = "splinef2d", bounds_error = False)

    #To be sure to plot all data
    z[np.where(np.isnan(z))] = 0.0

    if log_colors: z = np.log10(z)

    # Sort the points by density, so that the densest points are plotted last
    if sort :
        idx = z.argsort()
        x, y, z = x[idx], y[idx], z[idx]

    ax.scatter( x, y, c=z, rasterized=True, **kwargs)   # rasterize to save PDF file sizes

    # norm = Normalize(vmin = np.min(z), vmax = np.max(z))
    # cbar = fig.colorbar(cm.ScalarMappable(norm = norm), ax=ax)
    # cbar.ax.set_ylabel('Density')

    return ax

def format_log_log(ax):
    """
        Input:
            ax - plt.Axes
        Output:
            None

        Format plot where data has been log10'ed. Make all tick labels latex-formatted 10**(ticklabel)
        and only every multiple of 10. Make minorticks log spaced. If only one majortick, add majorticks at halfway points.
    """
    _format_ax_log(ax, True)
    _format_ax_log(ax, False)

    
def format_log_linear(ax):
    """
        Format xaxis logarithmically.
        Input:
            ax - plt.Axes
        Output:
            None

        Format plot where x data has been log10'ed. Make x tick labels latex-formatted 10**(ticklabel)
        and only every multiple of 10. Remove minor ticks as they are not log spaced.
    """
    _format_ax_log(ax, True)
def format_linear_log(ax):
    """
        Format yaxis logarithmically.
        Input:
            ax - plt.Axes
        Output:
            None

        Format plot where y data has been log10'ed. Make x tick labels latex-formatted 10**(ticklabel)
        and only every multiple of 10. Remove minor ticks as they are not log spaced.
    """
    _format_ax_log(ax, False)
def _format_ax_log(ax, x_values = True):

    if x_values: min_x, max_x = ax.get_xlim()
    else: min_x, max_x = ax.get_ylim()

    rotation = 0

    # Create major ticks
    majortick_locs_x = np.arange(int(min_x)-2, int(max_x)+2, dtype=float)
    # print(x_values, max_x, (max_x-min_x))
    if (max_x-min_x)>1.75:
        majortick_labels_x = np.array([f'${n2str.scientific_latex(10**x, 0)}$' for x in majortick_locs_x])
    else: # add majorticks at half integers
        for i in range(int(min_x)-2, int(max_x)+2): majortick_locs_x = np.append(majortick_locs_x, i+np.log10(3))
        if max_x>3:
            
            majortick_labels_x = np.array([f'${n2str.scientific_latex(10**x, 0)}$' if int(x)==x else f'${n2str.scientific_latex(10**x, 1)}$' for x in majortick_locs_x])
            if x_values: rotation = -70
        else:
            majortick_labels_x = np.array([f'${10**x:.0f}$' if int(x)==x else f'${10**x:.0f}$' for x in majortick_locs_x])\

    # Create log-spaced minorticks. 
    minortick_locs_x = []
    for i in range(int(min_x)-2, int(max_x)+2): minortick_locs_x.extend([i+n for n in np.log10(np.arange(1,10))])
    minortick_locs_x = np.array(minortick_locs_x)

    # Don't set any outside of bounds; this could change limits
    minortick_locs_x = minortick_locs_x[(minortick_locs_x>=min_x) & (minortick_locs_x<=max_x)]
    majortick_labels_x = majortick_labels_x[(majortick_locs_x>=min_x) & (majortick_locs_x<=max_x)]
    majortick_locs_x = majortick_locs_x[(majortick_locs_x>=min_x) & (majortick_locs_x<=max_x)]

    # Set tick locs

    if x_values: 
        ax.set_xticks(majortick_locs_x, minor=False)
        ax.set_xticklabels(majortick_labels_x, rotation=rotation)
        ax.set_xticks(minortick_locs_x, minor=True)
    else: 
        ax.set_yticks(majortick_locs_x, minor=False)
        ax.set_yticklabels(majortick_labels_x, rotation=rotation)
        ax.set_yticks(minortick_locs_x, minor=True)


def format_cloud_mask(ax):
    # Set aspect to equal, remove axes
    ax.set_aspect('equal')
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)

def set_text_size(font_small=8, font_large=10):
    plt.rc('font', size=font_small)          # controls default text sizes
    plt.rc('axes', titlesize=font_large)     # fontsize of the axes title
    plt.rc('axes', labelsize=font_small)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=font_small)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=font_small)    # fontsize of the tick labels
    plt.rc('legend', fontsize=font_small)    # legend fontsize
    plt.rc('figure', titlesize=font_large)  # fontsize of the figure title

def savefig(filename, pdf=False, dir=dirs.figures_directory, dpi=400, transparent=False, **kwargs): 
    # Save figure, but make filename filesystem-safe, either as pdf, svg or png with dpi
    filename = filename.replace(' ','_').replace('\n','_').replace(':','_').replace('/','_').replace('\\','').replace('{','').replace('}','').replace('textbf','')
    if pdf==True or pdf=='pdf': plt.savefig(dir+filename+'.pdf', dpi=dpi, transparent=transparent, **kwargs)   # still want dpi for rasterized graphics
    elif pdf=='svg': plt.savefig(dir+filename+'.svg', dpi=dpi, transparent=transparent, **kwargs)   # still want dpi for rasterized graphics
    else: plt.savefig(dir+filename+'.png', dpi=dpi, transparent=transparent, **kwargs)
    plt.close()
def ax_font_sizes(ax, small_size, large_size):
    # Set font sizes of all bits of text to small or large sizes (in px). Does not do legend.

    ax.xaxis.label.set_fontsize(small_size)
    ax.yaxis.label.set_fontsize(small_size)
    for label in ax.get_xticklabels(): label.set_fontsize(small_size)
    for label in ax.get_yticklabels(): label.set_fontsize(small_size)
    ax.title.set_fontsize(large_size)

# GIF:
def create_gif(plotting_function, to_plot, save_as, fps=10, dpi = 100, invert=False, **kwargs):
    """
        Input:
            plotting function: Function to plot with for ex, plt.pcolormesh. This function should create the figure.
            to_plot: List of things to plot. Each thing will be a frame in gif. 
                        Passed to plotting_function individually
            save_as: str: filename
            fps: int, frames per second,
            dpi: dpi of individual frames
            invert: Invert y-axis, for goes images
            **kwargs: Passed to plotting_function
        Out:
            None. Saves gif to figures_dir+save_as.

        Creates temporary folder with frames, saves images, reads and creates gif, then deletes temporary files.
    """
    # Create temp folder and save images to it:
    if os.path.exists(dirs.figures_directory+'__temp_for_gif__'): os.rmdir(dirs.figures_directory+'__temp_for_gif__')
    os.mkdir(dirs.figures_directory+'__temp_for_gif__')
    filenames = [dirs.figures_directory+'__temp_for_gif__/{}.png'.format(i) for i in range(len(to_plot))]
    for fname, arg in zip(filenames, to_plot):
        plotting_function(arg, **kwargs)
        plt.savefig(fname, dpi=dpi)
        plt.close()

    # Read and make gif:
    images = []
    for fname in filenames: images.append(imageio.imread(fname))
    imageio.mimsave(dirs.figures_directory+save_as+'.gif', images, duration=1000*1/fps, subrectangles=True)

    # Delete temp files:
    for fname in filenames: os.remove(fname)
    os.rmdir(dirs.figures_directory+'__temp_for_gif__')

if __name__ == '__main__':
    main()