import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # disable oneDNN custom operations

import warnings
warnings.filterwarnings('ignore')

import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)

import matplotlib.pyplot as plt
import typer
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.layers import Dense
from keras.losses import MeanSquaredError
from keras.models import Sequential
from keras.optimizers import Adam
from typing import Literal

from romgw.config.env import PROJECT_ROOT
from romgw.deep_learning.io import load_raw_training_data, save_scaler, load_scaler
from romgw.deep_learning.preprocessing import train_val_split, make_x_scaler, make_y_scaler
from romgw.typing.core import BBHSpinType, ModeType, ComponentType
from romgw.typing.utils import validate_literal

HELP_BBH_SPIN = 'Spin configuration: "NS" = no-spin, "AS" = aligned-spin, "PS" = precessing-spin.'
HELP_MODE = 'Spin-weighted spherical harmonic label (l, m): "2,2", "2,1", "3,2", "3,3", "4,4", or "4,3".'
HELP_COMPONENT = 'Waveform component: "amplitude" or "phase".'
HELP_MODEL_NAME = 'Name of the model: "NonLinRegV1"'
HELP_SAVING = "Whether to save redued basis and greedy errors."
HELP_VERBOSE = "Enable verbose output."

app = typer.Typer(help="Reduced Order Modelling of Gravitational Waves CLI")

@app.command()
def main(
    bbh_spin: BBHSpinType = typer.Option(..., help=HELP_BBH_SPIN),
    mode: ModeType = typer.Option(..., help=HELP_MODE),
    component: ComponentType = typer.Option(..., help=HELP_COMPONENT),
    model_name: Literal["NonLinRegV1"] = typer.Option(..., help=HELP_MODEL_NAME),
    saving: bool = typer.Option(False, "--saving/--no-saving", help=HELP_SAVING),
    verbose: bool = typer.Option(False, "--verbose/--no-verbose", help=HELP_VERBOSE),
) -> None:
    """
    Examples
    --------
    Train and save the NonLinRegV1 ANN for the amplitude of the 2,2 mode
    of the GW produced by the merger of an NS BBH:

        $ python train_ann.py --bbh-spin NS --mode 2,2 --component amplitude --model-name NonLinRegV1 --saving
    """
    bbh_spin = validate_literal(bbh_spin, BBHSpinType)
    mode = validate_literal(mode, ModeType)
    component = validate_literal(component, ComponentType)

    data_dir = PROJECT_ROOT / "data" / bbh_spin / "train" / mode / component
    if not data_dir.is_dir():
        raise NotADirectoryError(f"Could not find the directory {data_dir}")
    
    # ===== Preprocess data =====
    X_raw, Y_raw = load_raw_training_data(bbh_spin, mode, component)

    model_dir = data_dir / "models" / model_name
    model_dir.mkdir(parents=True, exist_ok=True)

    x_scaler = make_x_scaler(bbh_spin)
    y_scaler = make_y_scaler(Y_raw)

    save_scaler(x_scaler, "x", model_dir)
    save_scaler(y_scaler, "y", model_dir)

    x_scaler = load_scaler("x", model_dir)
    y_scaler = load_scaler("y", model_dir)

    X = x_scaler.transform(X_raw)
    Y = y_scaler.transform(Y_raw)

    X_train, X_val, Y_train, Y_val = train_val_split(X, Y)
    if verbose:
        typer.echo(f"{X_train.shape=}, {Y_train.shape=}")
        typer.echo(f"{X_val.shape=}, {Y_val.shape=}")

    # ===== Build model =====
    n_xfeatures = X_train.shape[1]
    n_yfeatures = Y_train.shape[1]

    if component == "amplitude":
        model = Sequential([
            Dense(n_xfeatures, activation='linear'),
            Dense(32, activation='relu'),
            Dense(64, activation='relu'),
            Dense(32, activation='sigmoid'),
            Dense(n_yfeatures, activation='linear')
        ])
    else:  # if component == "phase"
        model = Sequential([
            Dense(n_xfeatures, activation='linear'),
            Dense(128, activation='sigmoid'),
            Dense(n_yfeatures, activation='linear')
        ])

    # ----- Component-dependent kwargs -----
    optimiser_kwargs = {
        "amplitude": {
            "epsilon": 1e-6,
            "learning_rate": 3e-2
        },
        "phase": {
            "epsilon": 1e-6,
            "learning_rate": 1e-2
        }
    }
    lrop_kwargs = {
        "amplitude": {
            "factor": 0.5,
            "patience": 32,
            "min_lr": 1e-10
        },
        "phase": {
            "factor": 0.5,
            "patience": 32,
            "min_lr": 1e-10
        }
    }
    early_stopping_kwargs = {
        "amplitude": {
            "patience": 256
        },
        "phase": {
            "patience": 256
        }
    }
    fit_kwargs = {
        "amplitude": {
            "epochs": 1000,
            "batch_size": 64
        },
        "phase": {
            "epochs": 1000,
            "batch_size": 64
        }
    }

    # ----- Model optimisers/callbacks -----
    loss_fn = MeanSquaredError()

    optimiser = Adam(**optimiser_kwargs[component])

    lrop = ReduceLROnPlateau(monitor='val_loss', **lrop_kwargs[component])

    early_stopping = EarlyStopping(monitor='val_loss',
                                **early_stopping_kwargs[component],
                                restore_best_weights=True)

    model.compile(optimizer=optimiser, loss=loss_fn)

    # ----- Train the model -----
    history = model.fit(x=X_train,
                        y=Y_train,
                        validation_data=(X_val, Y_val),
                        **fit_kwargs[component],
                        callbacks=[lrop, early_stopping],
                        shuffle=True,
                        verbose=2)
    if verbose:
        typer.echo(f"Training of {model_name} complete.")

    # ----- Visualise gradient descent -----
    fig_file = model_dir / "training_curve.png"

    fig, ax = plt.subplots(1, 1)

    ax.loglog(history.history['val_loss'], label='val_loss')
    ax.loglog(history.history['loss'], label='loss')
    ax.loglog(history.history['learning_rate'], label='lr')

    ax.set_title(f"{model_name}: {bbh_spin}, {mode}, {component}")
    ax.set_xlabel("Epoch")
    ax.legend()
    fig.tight_layout()

    if saving:
        plt.savefig(fig_file)
    if saving and verbose:
        typer.echo(f"Saved training curve figure to {fig_file}")

    plt.show()

    # ===== Save model =====
    if saving:
        model_file = model_dir / "model.keras"
        model.save(filepath=model_file)
    if saving and verbose:
        typer.echo(f"Saved model to {model_file}")

if __name__ == "__main__":
    app()
