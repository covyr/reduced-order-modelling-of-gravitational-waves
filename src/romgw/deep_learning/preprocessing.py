import numpy as np
from pathlib import Path

from sklearn.preprocessing import FunctionTransformer, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from romgw.typing.core import RealArray, BBHSpinType
from romgw.typing.utils import validate_literal

def train_val_split(
    X: RealArray,
    Y: RealArray,
    training_frac: float = 0.8,
    seed: int | None = None
) -> tuple[RealArray, RealArray, RealArray, RealArray]:
    """
    Shuffle and split data into training and validation sets.

    Parameters
    ----------
    X: RealArray of shape (n_datapoints, n_xfeatures)
        Model input data to be split, consisting of the parameters associated
        with waveforms (`Y`).

    Y: RealArray of shape (n_datapoints, n_yfeatures)
        Model output data to be split, consisting of values at the empirical
        time nodes of waveforms, with which the parameters (`X`) are
        associated.

    training_frac: float
        Fraction of whole set to use for training. The remaining proportion
        is used for validation. Default value is 0.8.

    seed: int | None
        Used to seed random processes for reproducibility purposes. If no
        seed is provided, then the user foregoes reproducibility of
        random processes. Default value is None.

    Returns
    -------
    X_train: RealArray of shape (n_training_datapoints, n_xfeatures)

    X_validation: RealArray of shape (n_validation_datapoints, n_xfeatures)

    Y_train: RealArray of shape (n_training_datapoints, n_yfeatures)

    Y_validation: RealArray of shape (n_validation_datapoints, n_yfeatures)

    """
    # Combine X, Y data into one array.
    data = np.hstack([X, Y])

    # Sizes of data_train and data_test.
    train_size = int(training_frac * data.shape[0])

    # Indexes for data_train and data_test.
    rng = np.random.default_rng(seed=seed)
    idxs_permuted = rng.permutation(data.shape[0])
    
    # Split X and Y data into training and validation sets.
    X_train, X_validation = np.split(X[idxs_permuted, :], (train_size,), axis=0)
    Y_train, Y_validation = np.split(Y[idxs_permuted, :], (train_size,), axis=0)

    return X_train, X_validation, Y_train, Y_validation


def log_q(X: RealArray) -> RealArray:
    """"""
    X = np.asarray(X)
    X_is_1d = False

    if np.ndim(X) == 1:
        X_is_1d = True
        X = X[np.newaxis, :]

    X_out = np.log10(X)

    return X_out[0] if X_is_1d else X_out


def inverse_log_q(X: RealArray) -> RealArray:
    """"""
    X = np.asarray(X)
    X_is_1d = False

    if np.ndim(X) == 1:
        X_is_1d = True
        X = X[np.newaxis, :]

    X_out = 10**X
    return X_out[0] if X_is_1d else X_out


def make_x_scaler(
    bbh_spin: BBHSpinType,
) -> ColumnTransformer:
    """"""
    bbh_spin = validate_literal(bbh_spin, BBHSpinType)

    q_log_transformer = Pipeline([
        ('log', FunctionTransformer(log_q, inverse_log_q, validate=False)),
        ('scale', MinMaxScaler())
    ])
    preprocessor = ColumnTransformer(
        transformers=[
            ('q_log', q_log_transformer, [0]),
        ]
    )
    
    if bbh_spin == "NS":
        param_domain = np.array([[1.0],
                                 [10.0]], dtype=np.float64)
    elif bbh_spin == "AS":
        param_domain = np.array([[1.0, -1.0],
                                 [10.0, 1.0]], dtype=np.float64)
    else:  # if bbh_spin == "PS"
        param_domain = np.array([[1.0, -1.0, -1.0, -1.0],
                                 [10.0, 1.0, 1.0, 1.0]], dtype=np.float64)
        
    scaler = preprocessor.fit(param_domain)
    return scaler


def make_y_scaler(
    Y_raw: RealArray,
) -> MinMaxScaler:
    """"""
    preprocessor = MinMaxScaler()
    scaler = preprocessor.fit(Y_raw)
    return scaler
