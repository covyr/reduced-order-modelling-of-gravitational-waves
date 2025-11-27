import numpy as np

from romgw.waveform.base import (
    BaseWaveform,
    ComponentWaveform,
    FullWaveform
)

def rewrap_like(wf: BaseWaveform, new_wf_arr: np.ndarray) -> BaseWaveform:
    """Rewrap a new waveform array in the same waveform subclass."""
    if isinstance(wf, ComponentWaveform):
        return ComponentWaveform(new_wf_arr, wf.params, wf.component)
    elif isinstance(wf, FullWaveform):
        return FullWaveform(new_wf_arr, wf.params)
    else:
        return wf.__class__(new_wf_arr, wf.params)
