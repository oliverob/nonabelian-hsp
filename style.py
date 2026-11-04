"""Physical Review A figure style, shared by every plotting module.

Figures are drawn at their printed size in revtex4-2 two-column layout
(COLUMN for \\columnwidth, PAGE for \\textwidth), so the 8 pt lettering prints
at 8 pt. Text is Computer Modern, matching the manuscript. Axes are boxed,
with inward ticks on all four sides. Panels are labelled (a), (b), ... by
panel_labels(); a short heading (panel_headings()) may sit above
a panel, on the same baseline, and row_label() names a row of panels from
its left. Both are bold, to set them apart from axis labels (setting(), e.g.
b = 1, keeps the variable italic); the caption says the rest.
"""
from string import ascii_lowercase

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

COLUMN = 246/72.27   # revtex4-2 twocolumn \columnwidth, in inches.
PAGE = 510/72.27     # \textwidth: figure* environments.
COLORS = ["#0072B2", "#D55E00", "#009E73", "#CC79A7"]  # Okabe-Ito, safe for color-blind readers.
BOLD = "cmb10"       # Computer Modern bold, for panel headings and row labels.

RC = {
    "font.family": "serif", "font.serif": ["cmr10"], "mathtext.fontset": "cm",
    "axes.formatter.use_mathtext": True, "axes.unicode_minus": False,
    "font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "legend.fontsize": 7, "axes.titlesize": 8, "figure.labelsize": 8,
    "axes.linewidth": .5, "axes.spines.top": True, "axes.spines.right": True,
    "axes.prop_cycle": matplotlib.cycler(color=COLORS), "axes.labelpad": 2,
    "xtick.direction": "in", "ytick.direction": "in", "xtick.top": True, "ytick.right": True,
    "xtick.major.size": 3, "ytick.major.size": 3, "xtick.minor.size": 1.5, "ytick.minor.size": 1.5,
    "xtick.major.width": .5, "ytick.major.width": .5, "xtick.minor.width": .4, "ytick.minor.width": .4,
    "xtick.major.pad": 2.5, "ytick.major.pad": 2.5,
    "lines.linewidth": 1, "lines.markersize": 3.5, "lines.markeredgewidth": .6,
    "patch.linewidth": .5, "hatch.linewidth": .5, "errorbar.capsize": 0,
    "legend.frameon": False, "legend.handlelength": 1.8, "legend.handletextpad": .5,
    "legend.columnspacing": 1.2, "legend.borderaxespad": .3, "legend.labelspacing": .3,
    "figure.constrained_layout.h_pad": .02, "figure.constrained_layout.w_pad": .02,
    "savefig.dpi": 300, "savefig.pad_inches": .01, "pdf.fonttype": 42,
}


def apply():
    plt.rcParams.update(RC)


def above(ax, text, x, ha, **style):
    """Text just above the top edge of ax, at axes fraction x, on the shared baseline."""
    ax.annotate(text, (x, 1), xycoords="axes fraction", xytext=(0, 3.5), textcoords="offset points",
                ha=ha, va="baseline", fontsize=8, **style)


def setting(name, value):
    """A bold heading such as b = 1, with the variable in bold italic.

    Mathtext's \\mathbf does not embolden digits in Computer Modern, so only the
    variable is math; the rest is set in BOLD."""
    return rf"$\boldsymbol{{{name}}}$ = {value}"


def panel_labels(axes, start=0):
    """Label each axes (a), (b), ... just above its top-left corner, in reading order."""
    for index, ax in enumerate(axes):
        above(ax, f"({ascii_lowercase[start+index]})", 0, "left")


def row_label(ax, text):
    """A short label for a whole row of panels, set upright to the left of the row's
    first y label, so it does not read as part of that label."""
    ax.annotate(text, (0, .5), xycoords=ax.yaxis.label, xytext=(-4, 0), textcoords="offset points",
                ha="right", va="center", fontsize=8, family=BOLD)


def panel_headings(axes, headings):
    """A short bold heading, such as setting("b", 1), centered above each axes."""
    for ax, heading in zip(axes, headings):
        above(ax, heading, .5, "center", family=BOLD)
