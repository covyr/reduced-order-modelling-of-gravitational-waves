from matplotlib import gridspec
import matplotlib as mpl
import matplotlib.pyplot as plt

from romgw.config.constants import COMMON_TIME
from romgw.maths.core import mismatch
# from romgw.typing.core import ComponentType
from romgw.config.types import ComponentType
from romgw.waveform.base import FullWaveform
from romgw.waveform.dataset import FullWaveformDataset

from romgw.plotting.custom_config import RC_PARAMS
mpl.rcParams.update(RC_PARAMS)

def plot_same_mode(
    waveforms: FullWaveformDataset,
    verbose:bool = False,
) -> None:
    """Look into how parameters affect the fiducial waveforms."""
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
            hspace=0.3
        )

    axf = fig.add_subplot(spec[0])
    axa = fig.add_subplot(spec[1])
    axp = fig.add_subplot(spec[2])

    for wf in waveforms:
        axf.plot(COMMON_TIME, wf.real, label=wf.params)
        axa.plot(COMMON_TIME, wf.amplitude)
        axp.plot(COMMON_TIME, wf.phase)
        
    for ax, label in zip([axf, axa, axp], [r"$h$", r"$A$", r"$\phi$"]):
        ax.set_xlim(-5000, 250)
        ax.set_xlabel(r"$t$")
        ax.set_ylabel(label)

    if verbose:
        axl = fig.add_subplot(spec[3])
        axl.set_axis_off()
        handles, labels = axf.get_legend_handles_labels()
        axl.legend(handles, labels, loc='center', ncol=2, framealpha=0)
    
    plt.show()


def plot_same_params(
    modes: dict[str, FullWaveform],
) -> None:
    """Look into how the fiducial waveforms differ across modes."""
    fig = plt.figure(figsize=(12, 9 + 1/3))
    spec = gridspec.GridSpec(
        ncols=1,
        nrows=4,
        height_ratios=(3, 3, 3, 1/3),
        left=0,
        right=1,
        hspace=0.4
    )

    axf = fig.add_subplot(spec[0])
    axa = fig.add_subplot(spec[1])
    axp = fig.add_subplot(spec[2])

    for mode, wf in modes.items():
        axf.plot(COMMON_TIME, wf.real, label=f"{mode=}")
        axa.plot(COMMON_TIME, wf.amplitude)
        axp.plot(COMMON_TIME, wf.phase)
        
    for ax, label in zip([axf, axa, axp], [r"$h$", r"$A$", r"$\phi$"]):
        ax.set_xlim(-5000, 250)
        ax.set_xlabel(r"t")
        ax.set_ylabel(label)

    axl = fig.add_subplot(spec[3])
    axl.set_axis_off()
    handles, labels = axf.get_legend_handles_labels()
    axl.legend(handles, labels, loc='center', ncol=3, framealpha=0)

    plt.show()
    

def plot_mismatch(
    fiducial_waveforms: FullWaveformDataset,
    surrogate_waveforms: FullWaveformDataset,
    component: ComponentType | None = None,
) -> None:
    """Look into accuracy of reduced order model compared to fiducial."""
    n = len(fiducial_waveforms)

    for i in range(n):
        if not fiducial_waveforms[i].params == surrogate_waveforms[i].params:
            raise ValueError(f"Fiducial and surrogate are modelling"
                             f"different waveforms:\n"
                             f"fiducial : {fiducial_waveforms[i].params}\n"
                             f"surrogate: {surrogate_waveforms[i].params}")

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
        
        if not component:
            ylabel = r"$h$"
            wf_fid = fiducial_waveforms[i]
            wf_sur = surrogate_waveforms[i]
        elif component == "amplitude":
            ylabel = r"$A$"
            wf_fid = fiducial_waveforms[i].amplitude
            wf_sur = surrogate_waveforms[i].amplitude
        elif component == "phase":
            ylabel = r"$\phi$"
            wf_fid = fiducial_waveforms[i].phase
            wf_sur = surrogate_waveforms[i].phase

        ax.plot(COMMON_TIME, wf_fid.real, label="fiducial")
        ax.plot(COMMON_TIME, wf_sur.real, label="surrogate", linestyle='--')

        x_text = (xmin := ax.get_xlim()[0]) + 0.05*(ax.get_xlim()[1] - xmin)
        y_text = (ymin := ax.get_ylim()[0]) + 0.05*(ax.get_ylim()[1] - ymin)
        ax.text(x=x_text,
                y=y_text,
                s=rf"$1-\mathcal{{{'O'}}}\simeq{mismatch(wf_fid, wf_sur):.2e}$")
        
        ax.set_xlim(-5000, 250)
        ax.set_xlabel(r"$t$")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{fiducial_waveforms[i].params}")

    plt.show()
