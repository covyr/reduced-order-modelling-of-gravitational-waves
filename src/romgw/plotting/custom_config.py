import matplotlib as mpl

RC_PARAMS = {
    # Colours
    "figure.facecolor"  : "#0E191E", # Entire figure background
    "axes.facecolor"    : "#0E191E", # Plot (axes) background
    "savefig.facecolor" : "#0E191E", # Background in saved figures
    "axes.edgecolor"    : "white",
    "axes.labelcolor"   : "white",
    "xtick.color"       : "white",
    "ytick.color"       : "white",
    "text.color"        : "white",
    "axes.titlecolor"   : "white",
    "grid.color"        : "white", # Optional: faint grid
    
    # Other formatting
    "lines.linewidth"   : 1,
}
mpl.rcParams.update(RC_PARAMS)
