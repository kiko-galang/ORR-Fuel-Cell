import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import mpl_fontkit as fontkit
import sympy as sym
import scipy as sp
from scipy.stats import linregress

import seaborn as sns
import string

# Set plot parameters
params = {'mathtext.default': 'regular'}
plt.rcParams.update(params)

# Initialize Plotter
roman = ["i", "ii", "iii", "iv", "v", "vi", "vii", "viii", "ix", "x"]

# Set Plotter to Export SVG Vector Graphics
mpl.use('svg')
fontkit.install("Lato")
new_rc_params = {
    'text.usetex': False,
    'svg.fonttype': 'none',
    'font.family': 'sans-serif',
    'font.sans-serif': 'Lato',
}
mpl.rcParams.update(new_rc_params)

# Set Line Width of Axes
mpl.rcParams['axes.linewidth'] = 1

# --- Define all color palettes ---

## --- Color LookUp Function ---

def color_lookup(color,sheet):
    xls = pd.ExcelFile('Color Swatches from UC Berkeley.xlsx')
    df1 = pd.read_excel(xls, sheet)
    for i in range(0,len(df1['Color'])):
        if color == df1['Color'][i]:
            index = i
            return index
    return "Color Not Found"

def custom_palette(color_list,sheet):
    xls = pd.ExcelFile('Color Swatches from UC Berkeley.xlsx')
    df1 = pd.read_excel(xls, sheet)
    HEX_color = []
    for item in color_list:
        HEX_color.append("#"+str(df1['HEX'][color_lookup(item,sheet)]))
    return sns.color_palette(HEX_color)

tableau20 = [
    (31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120),
    (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),
    (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),
    (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),
    (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229)
]
tableau20 = [(e[0] / 255.0, e[1] / 255.0, e[2] / 255.0) for e in tableau20]
colors = sns.color_palette(tableau20)

barplot = [
    (59., 195., 113.), (19., 116., 157.), (36., 170., 227.), (147., 183., 219.),
    (162., 0., 0.), (255., 68., 68.), (202., 102., 40.), (248., 153., 58.)
]
barplot = [(e[0] / 255.0, e[1] / 255.0, e[2] / 255.0) for e in barplot]
rainbow_2 = sns.color_palette(barplot)

j_i_pal = [
    (250., 172., 116.), (194., 104., 47.), (247., 168., 170.), (240., 79., 82.),
    (160., 191., 216.), (98., 146., 190.), (64., 105., 149.), (166., 91., 164.),
    (133., 127., 188.), (66., 69., 133.), (65., 67., 76.), (184., 184., 187.)
]
j_i_pal = [(e[0] / 255.0, e[1] / 255.0, e[2] / 255.0) for e in j_i_pal]
rainbow_1 = sns.color_palette(j_i_pal)

# More color palettes
xls = pd.ExcelFile('Color Swatches from UC Berkeley.xlsx')

df1 = pd.read_excel(xls, 'LEGO')
color_list = ["Bright Light Blue", 'Dark Blue','Sand Red',"Dark Red",'Medium Bluish Violet',
              "Dark Purple","Light Orange","Dark Orange","Sand Green", "Dark Green"]
rainbow_4 = custom_palette(color_list,"LEGO")

df1 = pd.read_excel(xls, 'PrismaColor')
PrismaColor = ["#" + str(df1['HEX'][i]) for i in range(len(df1['Color']))]
PrismaColor = sns.color_palette(PrismaColor)

df1 = pd.read_excel(xls, 'Pantone')
Pantone = ["#" + str(df1['HEX'][i]) for i in range(len(df1['Color']))]
Pantone = sns.color_palette(Pantone)

color_list = ["290 C", "291 C", "292 C", "293 C", "294 C", "295 C", "296 C"]
Pantone_Blue = custom_palette(color_list, "Pantone")

color_list = ["169 C", "170 C", "171 C", "172 C", "173 C", "174 C", "175 C"]
Pantone_Red = custom_palette(color_list, "Pantone")

color_list = ["296 C", "295 C", "294 C", "293 C", "292 C", "291 C", "290 C",
              "169 C", "170 C", "171 C", "172 C", "173 C", "174 C", "175 C"]
Pantone_Red_Blue = custom_palette(color_list, "Pantone")


barplot2 = [(255.,255.,217.), (237.,248.,177.), (199.,233.,180.), (127.,205.,187.),    
             (65.,182.,196.), (29.,145.,192.), (34.,94.,168.), (37.,52.,148.), (8.,29.,88.)] 

barplot2 = [(e[0] / 255.0, e[1] / 255.0, e[2] / 255.0) for e in barplot2]

cool_sequential = (sns.color_palette(barplot2))

barplot2 = [(255., 246., 204.),(250., 229., 136.), (254.,192.,49.), (247.,143.,30.), (232., 93., 4.),(228., 77., 46.),(203., 11., 10.), (157., 2., 8.), (106., 4., 15.)] 

barplot2 = [(e[0] / 255.0, e[1] / 255.0, e[2] / 255.0) for e in barplot2]

warm_sequential = (sns.color_palette(barplot2))

# Print available color palettes
print("tableau20, barplot, rainbow_1, rainbow_2, j_i_pal, rainbow_4, PrismaColor, Pantone, Pantone_Blue, Pantone_Red, Pantone_Red_Blue")

## --- Plotting Functionality ---

def gengrid(
    n_cols=1, n_rows=1, dpi_fig=600, fig_size=(8, 6), genlabels=True,
    size_inches=(3.25, 2.5), ticklabel_size=8, label_pos=-0.15,
    bold=False, minor=True, secondary_yaxis=None
):
    # Generate Figure Dimensions
    fig, axs = plt.subplots(ncols=n_cols, nrows=n_rows, dpi=dpi_fig, figsize=fig_size)
    sec_ax_list = [[None for _ in range(n_cols)] for _ in range(n_rows)]
    # Set plot size in inches
    fig.set_size_inches(size_inches[0], size_inches[1])

    # Check if Bold Text is Chosen
    bold_val = "bold" if bold else "regular"

    # Determine size of axs matrix
    try:
        x = np.shape(axs)[0]
    except:
        x = 0
    try:
        y = np.shape(axs)[1]
    except:
        y = 0

    def configure_axis(ax):
        ax.axes.tick_params(axis="x", which='both', direction="in", top=True, labelsize=ticklabel_size)
        ax.axes.tick_params(axis="y", which='both', direction="in", right=True, labelsize=ticklabel_size)
        if minor:
            ax.minorticks_on()
            ax.xaxis.set_minor_locator(mpl.ticker.AutoMinorLocator(2))
            ax.yaxis.set_minor_locator(mpl.ticker.AutoMinorLocator(2))
        else:
            ax.minorticks_off()
        for tick in ax.xaxis.get_major_ticks():
            tick.label1.set_fontsize(ticklabel_size)
            tick.label1.set_fontweight(bold_val)
        for tick in ax.yaxis.get_major_ticks():
            tick.label1.set_fontsize(ticklabel_size)
            tick.label1.set_fontweight(bold_val)
        [s.set_linewidth(1.25) for s in ax.spines.values()]

    def configure_secondary_axis(ax):
        sec_ax = ax.twinx()
        sec_ax.tick_params(axis="y", which='both', direction="in", labelsize=ticklabel_size)
        for tick in sec_ax.yaxis.get_major_ticks():
            tick.label1.set_fontsize(ticklabel_size)
            tick.label1.set_fontweight(bold_val)
        if minor:
            sec_ax.minorticks_on()
            sec_ax.yaxis.set_minor_locator(mpl.ticker.AutoMinorLocator(2))
        return sec_ax

    # Configure 2D grid
    if x > 1 and y > 1:
        for row in axs:
            for ax in row:
                configure_axis(ax)
        if secondary_yaxis:
            for idx in secondary_yaxis:
                row, col = idx
                sec_ax_list[row][col] = (configure_secondary_axis(axs[row][col]))
    # Configure 1D grids
    elif (x > 1 and not y > 1) or (x > 1 and not y > 1):
        for ax in axs:
            configure_axis(ax)
        if secondary_yaxis:
            for idx in secondary_yaxis:
                sec_ax_list[idx]=(configure_secondary_axis(axs[idx]))
    # Configure single panel
    else:
        configure_axis(axs)
        if secondary_yaxis:
            sec_ax_list = (configure_secondary_axis(axs))

    # Generate subplot labels
    if genlabels:
        alpha = list(string.ascii_lowercase)
        labels = np.array(alpha[0:np.size(axs)])
        labels = np.reshape(labels, np.shape(axs))
        if x > 1 and y > 1:
            for i in range(x):
                for j in range(y):
                    axs[i][j].text(
                        label_pos, 1.1, labels[i][j] + ")", transform=axs[i][j].transAxes,
                        fontsize=10, fontweight=bold_val, va='top', ha='right'
                    )
        elif (x > 1 and not y > 1) or (x > 1 and not y > 1):
            for j in range(x):
                axs[j].text(
                    label_pos, 1.1, labels[j] + ")", transform=axs[j].transAxes,
                    fontsize=10, fontweight=bold_val, va='top', ha='right'
                )
    return fig, axs, sec_ax_list

## Print Input Arguments for gengrid
print(gengrid.__code__.co_varnames)
