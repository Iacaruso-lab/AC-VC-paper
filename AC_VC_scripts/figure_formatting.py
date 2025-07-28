import matplotlib
from matplotlib import rcParams
import matplotlib.font_manager as fm
import pandas as pd
import sys
from matplotlib.lines import Line2D
import colorsys


def set_font_params(gen_parameters):
    rcParams["font.family"] = gen_parameters["font"]
    rcParams["font.size"] = gen_parameters["font_size"]
    rcParams["svg.fonttype"] = "none"
    rcParams["pdf.fonttype"] = 42
    rcParams["text.usetex"] = False


def print_sys():
    print(sys.executable, flush=True)


def myPlotSettings_splitAxis(fig, ax, ytitle, xtitle, title, axisColor="k", mySize=7):
    ax.spines["left"].set_color(axisColor)
    ax.spines["bottom"].set_color(axisColor)
    ax.xaxis.label.set_color(axisColor)
    ax.yaxis.label.set_color(axisColor)
    ax.tick_params(axis="x", colors=axisColor)
    ax.tick_params(axis="y", colors=axisColor)

    rcParams["font.size"] = mySize
    ax.set_ylabel(ytitle, fontsize=mySize, labelpad=1)
    ax.set_xlabel(xtitle, fontsize=mySize)
    ax.set_title(title, fontsize=mySize, weight="bold")
    for tick in ax.get_xticklabels():
        tick.set_fontsize(mySize)
    for tick in ax.get_yticklabels():
        tick.set_fontsize(mySize)
    right = ax.spines["right"]
    right.set_visible(False)
    top = ax.spines["top"]
    top.set_visible(False)
    ax.tick_params(width=0.25)
    for line in ["left", "bottom"]:
        ax.spines[line].set_linewidth(0.25)
        ax.spines[line].set_position(("outward", 3))


def convert_to_exp(num, sig_fig=2):
    mant, exp = f"{num:.{sig_fig}E}".split("E")
    if round(float(mant)) == 1:
        exp_add = exp
        symbol_add = "="
    else:
        exp_add = int(exp) + 1
        symbol_add = "<"
    return symbol_add, exp_add


def plot_errorbars(ax, distances, means, errors, color_map):
    for area in means.index:
        ax.errorbar(
            distances[area],
            means[area],
            yerr=errors[area],
            fmt="o",
            color="black",
            mfc=color_map.get(area, "lightgray"),
            mec=color_map.get(area, "lightgray"),
            ms=2,
            elinewidth=0.5,
            capsize=0.5,
        )


def style_figure_exp_fits(fig, axes, color_map, convert_dict, font_size):
    axes[0].set_ylabel("Projection probability", fontsize=font_size)
    axes[1].set_ylabel("Normalised projection density", fontsize=font_size)
    for ax in axes:
        ax.set_xticks([0, 2000, 4000])

    labels = [convert_dict.get(a, a) for a in color_map]
    dummy = [Line2D([], [], ls="none") for _ in labels]
    legend = fig.legend(
        dummy,
        labels,
        loc="center left",
        bbox_to_anchor=(0.4, 0.5),
        frameon=False,
        handlelength=0,
        handletextpad=0.1,
        fontsize=font_size,
        prop={"family": "Arial"},
    )
    for txt, area in zip(legend.get_texts(), color_map):
        txt.set_color(color_map[area])


def adjust_color(color, amount=1.0):
    """function to change the change the lightness of a specified color in the plot"""
    try:
        c = matplotlib.colors.cnames[color]
    except:
        c = color
    r, g, b = matplotlib.colors.to_rgb(c)
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    l = max(min(l * amount, 1.0), 0.0)
    r, g, b = colorsys.hls_to_rgb(h, l, s)
    return (r, g, b)
