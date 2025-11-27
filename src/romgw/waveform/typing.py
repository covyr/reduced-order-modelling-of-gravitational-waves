from typing import TypeVar

from romgw.waveform.base import BaseWaveform


WaveformT = TypeVar("WaveformT", bound=BaseWaveform)
