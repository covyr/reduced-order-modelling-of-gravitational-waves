from __future__ import annotations

import numpy as np

from romgw.config.env import COMMON_TIME, ZERO_TOL
# from romgw.maths.core import dot, mag2, normalise
from romgw.waveform.base import ComponentWaveform
from romgw.waveform.dataset import ComponentWaveformDataset
from romgw.waveform.utils import rewrap_like
from romgw.typing.core import RealArray


def dot(
    wf_a: ComponentWaveform,
    wf_b: ComponentWaveform,
    time: RealArray = COMMON_TIME,
) -> float:
    """
    `romgw.maths.core.dot()`, optimised for `ComponentWaveform`
    instances.
    """
    return float(np.trapezoid(np.asarray(wf_a) * np.asarray(wf_b), time))


def mag2(wf: ComponentWaveform, time: RealArray = COMMON_TIME) -> float:
    """
    `romgw.maths.core.mag2()`, optimised for `ComponentWaveform`
    instances.
    """
    return float(np.trapezoid(np.asarray(wf)**2, time))


def normalise(
    wf: ComponentWaveform,
    time: RealArray = COMMON_TIME,
    zero_tol: float = ZERO_TOL,
) -> ComponentWaveform:
    """
    `romgw.maths.core.normalise()`, optimised for `ComponentWaveform`
    instances.
    """
    norm_factor = np.sqrt(np.trapezoid(np.asarray(wf)**2, time))
    if norm_factor < zero_tol:
        raise ZeroDivisionError("Cannot normalise a zero waveform.")
    return rewrap_like(wf, np.asarray(wf) / norm_factor)


def gram_matrix(
    rb: ComponentWaveformDataset,
    time: RealArray = COMMON_TIME,
) -> RealArray:
    """Gram matrix check for a reduced basis."""
    m = len(rb)
    G = np.zeros((m, m), dtype=float)
    for i in range(m):
        for j in range(m):
            G[i, j] = float(dot(rb[i], rb[j], time))  # real-valued
    return G


def reduced_basis(
    waveforms: ComponentWaveformDataset,
    tolerance: float,
    time: RealArray = COMMON_TIME,
    max_basis: int | None  = None,
    zero_tol: float = ZERO_TOL,
    verbose: bool = False,
) -> tuple[ComponentWaveformDataset, RealArray]:
    """
    Construct an orthonormal reduced basis using a greedy algorithm
    with Modified Gram-Schmidt orthogonalisation.

    This function incrementally builds a reduced basis by iteratively
    selecting the waveform with the largest residual norm, orthonormalising
    it against the existing basis, and updating all residuals. The process
    continues until the maximum residual norm falls below the specified
    tolerance or the basis reaches the maximum allowed size.

    Parameters
    ----------
    waveforms : ComponentWaveformDataset
        Collection of input waveforms. The input object is not modified.
    tolerance : float
        Stopping criterion. The algorithm terminates when the maximum
        residual norm is less than or equal to this value.
    time : ndarray of shape (n_samples,), optional
        Time array used for computing inner products.
        Default is `COMMON_TIME`.
    max_basis : int or None, optional
        Maximum number of basis elements. If None, defaults to the total
        number of waveforms in the input collection. Default is None.
    zero_tol : float, optional
        Numerical tolerance for treating norms as zero to avoid
        round-off issues. Default is `ZERO_TOL`.
    verbose : bool, optional
        If True, prints diagnostic information and intermediate
        Gram matrices during construction. Default is False.

    Returns
    -------
    rb : ComponentWaveformDataset
        Orthonormal reduced basis constructed from the input waveforms.
    greedy_errors_arr : ndarray of shape (n_basis,)
        Array containing the maximum residual norm at each greedy
        iteration, tracking the approximation error as the basis grows.

    Notes
    -----
    The algorithm proceeds as follows:

    1. Initialize residuals as copies of the input waveforms.
    2. While `max(||residuals||) > tolerance` and `len(basis) < max_basis`:

       a. Select the residual with the largest norm.  
       b. Normalize it to create the new basis element `e_k`.  
       c. Append `e_k` to the reduced basis.  
       d. Update all residuals:

          .. math::

             r_j \\leftarrow r_j - \\langle r_j, e_k \\rangle e_k

    The Modified Gram-Schmidt procedure ensures numerical stability
    during the orthonormalization process.
    """
    # Defensive copy of residuals (create independent waveform instances).
    residuals: list[ComponentWaveform] = [
        rewrap_like(wf, wf.copy())
        for wf in waveforms._waveforms
    ]

    # Extract component value from residuals.
    component = str(residuals[1].component)

    # For filename/printing purposes -> all numbers have the same length.
    lsM = len(str((M := len(residuals)) - 1))

    if M == 0:
        return ComponentWaveformDataset(), np.array([], dtype=np.float64)

    if max_basis is None:
        max_basis = M

    # Initialise greedy algorithm.
    rb = ComponentWaveformDataset([], component=component)
    greedy_errors: list[float] = []

    # Initial residual norms.
    errors = np.array([mag2(r, time) for r in residuals])

    # TEMP: TESTING WHETHER NORMALISING AT THE START IMPROVES ORTHONORMALITY
    temp = np.argmax(errors)
    errors = np.concatenate((errors[:temp], np.array([1.0]), errors[temp+1:]))
    residuals: list[ComponentWaveform] = [
        rewrap_like(wf, normalise(wf.copy(), time, zero_tol))
        for wf in waveforms._waveforms
    ]

    # Greedy algorithm.
    while True:
        idx = int(np.argmax(errors))
        max_err = float(errors[idx])
        greedy_errors.append(max_err)

        if max_err <= tolerance:
            if verbose:
                print(' ' * 80, end='\r')
                print(f"Tolerance met ({max_err:.2e} <= {tolerance:.2e}) "
                      f"with {len(rb)} reduced basis elements.")
            break

        if len(rb) >= max_basis:
            if verbose:
                print(' ' * 80, end='\r')
                print(f"Tolerance met ({max_err:.2e} > {tolerance:.2e}) "
                      f"with {max_basis} reduced basis elements.")

        # Select residual with largest norm.
        r = residuals.pop(idx)
        errors = np.delete(errors, idx)

        # Skip if zero (numerically).
        if mag2(r, time) <= zero_tol:
            continue

        # Normalise -> new basis element.
        try:
            e = normalise(r, time, zero_tol)
        except ValueError:
            continue

        rb.add_waveform(e)

        # Update residuals: r_j <- r_j - <r_j, e> e
        # Stack all residuals into a matrix for vectorization
        R = np.vstack([np.asarray(r) for r in residuals])  # shape (n_residual, L)
        e_arr = np.asarray(e, dtype=np.float64)  # shape (L,)

        # Compute projection coefficients in one go: <R, e>
        # dot_row = integral conj(R_i) * e  over time.
        integrand = R * e_arr  # real-valued
        proj_coeffs = np.asarray(np.trapezoid(integrand, x=time, axis=1))

        # R <- R - coeffs[:,None] * e
        R = R - proj_coeffs[:, None] * e_arr
        # Push results back to ComponentWaveform objects.
        residuals = [rewrap_like(residuals[i], R[i])
                     for i in range(len(residuals))]
        errors = np.sum(R**2, axis=1)  # real-valued

        if verbose:
            print(f"m={len(rb):0{lsM}d}/{M}, "
                  f"greedy error={max_err:.2e}", end='\r')

    greedy_errors[0] = 1.0  # by definition
    greedy_errors_arr = np.array(greedy_errors, dtype=np.float64)

    return rb, greedy_errors_arr

def mgs(waveforms: ComponentWaveformDataset) -> ComponentWaveformDataset:
    """"""
    if len(waveforms) == 0:
        raise ValueError("No waveforms to orthonormalise.")
    
    # Defensive copy of residuals (create independent waveform instances).
    residuals: list[ComponentWaveform] = [
        rewrap_like(wf, wf.copy())
        for wf in waveforms._waveforms
    ]
    
    # Extract component value from residuals.
    component = str(residuals[0].component)

    orthonormed: list[ComponentWaveform] = [normalise(residuals.pop(0))]
    while True:
        if len(residuals) == 0:
            return ComponentWaveformDataset(orthonormed, component=component)

        for i in range(len(residuals)):
            residual_arr = (
                residuals[i] - (dot(residuals[i], orthonormed[-1])
                                * orthonormed[-1]
                                / mag2(orthonormed[-1])))
            residuals[i] = rewrap_like(residuals[i], residual_arr)

        orthonormed.append(normalise(residuals.pop(0)))
    