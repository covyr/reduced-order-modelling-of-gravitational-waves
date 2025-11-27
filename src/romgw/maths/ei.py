from typing import Tuple
import numpy as np

from romgw.typing.core import RealArray
from romgw.waveform.dataset import ComponentWaveformDataset


def empirical_time_nodes(
    rb: ComponentWaveformDataset
) -> Tuple[RealArray, RealArray]:
    """
    Compute the empirical time nodes for a reduced basis.

    This algorithm iteratively selects empirical interpolation points
    (time nodes) that best capture the behaviour of a reduced basis.
    At each iteration, the next node is chosen greedily as the index
    with the largest residual between the current basis vector and its
    empirical interpolant.

    Parameters
    ----------
    rb : WaveformCollection
        Collection of `m` orthonormal reduced-basis waveforms, each of
        length `L`.

    Returns
    -------
    etns_arr : (m,) ndarray of int
        Indices of the selected empirical time nodes, ordered by selection
        step.
    B : (L, m) ndarray of float
        Empirical interpolation matrix constructed from the selected nodes.

    Notes
    -----
    The algorithm proceeds as follows:

    1. Select the initial node, corresponding to the largest absolute value
       in the first basis vector.
    2. Iteratively select the node that maximizes the residual between
       each successive basis vector and its current empirical interpolant.
    3. Update the interpolation matrix ``B`` and its inverse submatrix
       at each step.

    This implementation assumes that the reduced basis is orthonormal and
    represented as a 2D array of shape `(m, L)` accessible via `rb.array`.
    """
    # ----- Initialise objects for greedy algorithm -----
    rb_arr = rb.array
    m, L = rb_arr.shape
    etns: list[int] = []

    # ----- Select 0-th node -----
    v0 = rb_arr[0].copy()  # 0-th rb element
    I = np.zeros_like(v0)  # no B matrix to find the empirical interpolant yet
    r0 = I - v0  # minimise this (aiming for I ~= v)
    
    # The 0-th node is unique by definition
    node0 = np.argmax(np.abs(r0))  # node which most needs additional 'help'
    etns.append(node0.astype(int))

    V_inv = np.linalg.inv(rb_arr[:1, etns].T)  # init V_inv with 0-th node
    B = rb_arr[:1, :].T @ V_inv  # init B with 0-th node

    # ----- Greedily select k-th nodes -----
    for k in range(1, m):
        v = rb_arr[k].copy()  # k-th rb element
        I = B @ v[etns]  # empirical interpolant
        r = I - v  # minimise this (aiming for I ~= v)
        
        r[etns] *= 0  # ensure unique node selection
        node = np.argmax(np.abs(r))  # node which most needs additional 'help'
        etns.append(node.astype(int))

        V_inv = np.linalg.inv(rb_arr[:k+1, etns].T)  # find updated V_inv
        B = rb_arr[:k+1, :].T @ V_inv  # find updated B

    etns_arr = np.array(etns)

    return etns_arr, B


def empirical_interpolant(h_nodes: np.ndarray, B: np.ndarray):
    """
    Reconstruct a waveform using the Empirical Interpolation Method (EIM).

    Given waveform values sampled at the empirical time nodes, this function
    reconstructs (interpolates) the full waveform using the precomputed
    empirical interpolation matrix ``B``.

    Parameters
    ----------
    h_nodes : (m,) ndarray
        Waveform values evaluated at the empirical time nodes.
    B : (L, m) ndarray
        Empirical interpolation matrix computed from the reduced basis,
        satisfying the approximation `h(t) ~= B(t) @ h(nodes)`.

    Returns
    -------
    I : (L,) ndarray
        Empirical interpolant representing the reconstructed full waveform.

    Notes
    -----
    The Empirical Interpolation Method (EIM) approximates a function
    ``h(t)`` using only a subset of its values at optimally chosen nodes:

    .. math::

        h(t) \\approx \\sum_{i=1}^{m} B_i(t) \\, h(t_i)

    where ``B`` contains the interpolation basis functions constructed
    from the reduced basis and the empirical time nodes.
    """
    return B @ h_nodes
