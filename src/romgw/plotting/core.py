from matplotlib import gridspec
import matplotlib as mpl
import matplotlib.pyplot as plt

from romgw.config.env import COMMON_TIME
from romgw.maths.core import mismatch
from romgw.waveform.base import FullWaveform
from romgw.waveform.dataset import FullWaveformDataset


mpl.rcParams.update({
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
})


def plot_same_mode(
    waveforms: FullWaveformDataset,
    verbose:bool = False,
) -> None:
    """Look into how the parameters affect the waveforms."""
    if isinstance(waveforms, FullWaveform):
        waveforms = [waveforms]
    elif not isinstance(waveforms, FullWaveformDataset):
        raise ValueError(f"Waveforms must be an instance of "
                         f"FullWaveform or FullwaveformDataset.")

    n = len(waveforms)

    if verbose:  # show legend with waveform params
        fig = plt.figure(figsize=(12, 9 + (n//2 + 1)/6))    
        spec = gridspec.GridSpec(
            ncols=1,
            nrows=4,
            height_ratios=(3, 3, 3, (n//2 + 1)/6),
            left=0,
            right=1,
            hspace=0.6
        )
    else:
        fig = plt.figure(figsize=(12, 3*3))
        spec = gridspec.GridSpec(
            ncols=1,
            nrows=3,
            height_ratios=(3, 3, 3),
            left=0,
            right=1,
            hspace=0.6
        )

    axf = fig.add_subplot(spec[0])
    axa = fig.add_subplot(spec[1])
    axp = fig.add_subplot(spec[2])

    for wf in waveforms:
        axf.plot(COMMON_TIME, wf.real, label=wf.params)
        axa.plot(COMMON_TIME, wf.amplitude)
        axp.plot(COMMON_TIME, wf.phase)
        
    for ax, label in zip([axf, axa, axp], ["", " amplitude", " phase"]):
        ax.set_xlim(-5000, 250)
        ax.set_xlabel("time")
        ax.set_ylabel(f"waveform{label}")
        ax.set_title(f"{n} waveform{label}" + ("s" if n > 1 else ""))

    if verbose:
        axl = fig.add_subplot(spec[3])
        axl.set_axis_off()
        handles, labels = axf.get_legend_handles_labels()
        axl.legend(handles, labels, loc='center', ncol=2, framealpha=0)

    plt.show()


def plot_same_params(
    modes: dict[str, FullWaveform],
) -> None:
    """Look into how the waveforms differ across the modes."""
    fig = plt.figure(figsize=(12, 9 + 1/3))
    spec = gridspec.GridSpec(
        ncols=1,
        nrows=4,
        height_ratios=(3, 3, 3, 1/3),
        left=0,
        right=1,
        hspace=0.6
    )

    axf = fig.add_subplot(spec[0])
    axa = fig.add_subplot(spec[1])
    axp = fig.add_subplot(spec[2])

    for mode, wf in modes.items():
        axf.plot(COMMON_TIME, wf.real, label=f"{mode=}")
        axa.plot(COMMON_TIME, wf.amplitude)
        axp.plot(COMMON_TIME, wf.phase)
        
    for ax, label in zip([axf, axa, axp], ["", " amplitude", " phase"]):
        ax.set_xlim(-5000, 250)
        ax.set_xlabel("time")
        ax.set_ylabel(f"waveform{label}")
        ax.set_title(f"waveform mode{label}s for {modes['2,2'].params}")

    axl = fig.add_subplot(spec[3])
    axl.set_axis_off()
    handles, labels = axf.get_legend_handles_labels()
    axl.legend(handles, labels, loc='center', ncol=3, framealpha=0)

    plt.show()
    

def plot_mismatch(
    fiducial_waveforms: FullWaveformDataset,
    surrogate_waveforms: FullWaveformDataset,
) -> None:
    """Compare fiducial waveforms to their surrogate counterparts."""
    n = len(fiducial_waveforms)

    fig = plt.figure(figsize=(12, 3*n))
    spec = gridspec.GridSpec(
        ncols=1,
        nrows=n,
        height_ratios=(*[3 for _ in range(n)],),
        left=0,
        right=1,
        hspace=0.6
    )

    axs = []

    for i in range(n):
        ax = fig.add_subplot(spec[i])
        axs.append(ax)
        
        wf_fid = fiducial_waveforms[i].real
        wf_sur = surrogate_waveforms[i].real

        ax.plot(COMMON_TIME, wf_fid, label="fiducial")
        ax.plot(COMMON_TIME, wf_sur, label="surrogate", linestyle='--')

        x_text = (xmin := ax.get_xlim()[0]) + 0.05*(ax.get_xlim()[1] - xmin)
        y_text = (ymin := ax.get_ylim()[0]) + 0.05*(ax.get_ylim()[1] - ymin)
        ax.text(x=x_text,
                y=y_text,
                s=f"mismatch={mismatch(wf_fid, wf_sur):.6e}")
        
        ax.set_xlim(-5000, 250)
        ax.set_xlabel("time")
        ax.set_ylabel(f"full waveform")
        ax.set_title(f"{fiducial_waveforms[i].params}")

    plt.show()
