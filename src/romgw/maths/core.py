import numpy as np
import lal
from pycbc import filter as pyfilter
from pycbc import psd as pypsd
from pycbc.types import TimeSeries

from romgw.config.env import COMMON_TIME, ZERO_TOL
from romgw.waveform.utils import rewrap_like
from romgw.waveform.typing import WaveformT

from romgw.waveform.base import (
    BaseWaveform,
    ComponentWaveform,
    FullWaveform,
)
from romgw.typing.core import (
    RealArray
)


def _dot_integrand(wf_a: BaseWaveform, wf_b: BaseWaveform) -> RealArray:
    """Return the real-valued integrand array for dot products."""
    a_arr = np.asarray(wf_a)
    b_arr = np.asarray(wf_b)
    if isinstance(wf_a, FullWaveform):
        integrand = a_arr.conjugate() * b_arr
    elif isinstance(wf_a, ComponentWaveform):
        integrand = a_arr * b_arr
    else:
        raise TypeError(f"Unsupported waveform type: {type(wf_a)}")
    return np.real(integrand).astype(np.float64, copy=False)


def dot(
    wf_a: BaseWaveform,
    wf_b: BaseWaveform,
    time: RealArray = COMMON_TIME,
) -> float:
    """Return the (possibly Hermitian) dot product of two waveforms."""
    return float(np.trapezoid(_dot_integrand(wf_a, wf_b), time))


def mag(wf: BaseWaveform, time: RealArray = COMMON_TIME) -> float:
    """Return the L2 norm (magnitude) of a waveform."""
    integrand = _dot_integrand(wf, wf)
    return float(np.sqrt(np.trapezoid(integrand, time)))


def mag2(wf: BaseWaveform, time: RealArray = COMMON_TIME) -> float:
    integrand = _dot_integrand(wf, wf)
    return float(np.trapezoid(integrand, time))


def normalise(
    wf: WaveformT,
    time: RealArray = COMMON_TIME,
    zero_tol: float = ZERO_TOL,
) -> WaveformT:
    """Return a normalised version of the waveform."""
    integrand = _dot_integrand(wf, wf)
    norm_factor = np.sqrt(np.trapezoid(integrand, time))
    if norm_factor < zero_tol:
        raise ZeroDivisionError("Cannot normalise a zero waveform.")
    return rewrap_like(wf, np.asarray(wf) / norm_factor)


def mismatch(h1: WaveformT, h2: WaveformT, M_sol:int = 50) -> float:
    """Return the mismatch between two waveforms h1 and h2."""

    # Sample time.
    dt = M_sol * lal.MTSUN_SI   # seconds

    # Make both arrays equal length (zero pad).
    N = max(len(h1), len(h2))
    if len(h1) < N:
        h1 = np.pad(h1, (0, N - len(h1)))
    if len(h2) < N:
        h2 = np.pad(h2, (0, N - len(h2)))

    # Construct TimeSeries objects.
    h1_ts = TimeSeries(h1.real, delta_t=dt)
    h2_ts = TimeSeries(h2.real, delta_t=dt)
    
    # PSD for aLIGO design sensitivity.
    delta_f = 1.0 / h1_ts.duration
    f_len = N // 2 + 1
    f_min = 20.0
    # f_max = 4096.0
    f_max = 2048.0

    psd = pypsd.aLIGODesignSensitivityP1200087(f_len, delta_f, f_min)

    # Compute the overlap.
    overlap, _ = pyfilter.match(
        h1_ts,
        h2_ts,
        psd=psd,
        low_frequency_cutoff=f_min,
        high_frequency_cutoff=f_max,
        subsample_interpolation=True,
    )

    return float(1 - overlap)
