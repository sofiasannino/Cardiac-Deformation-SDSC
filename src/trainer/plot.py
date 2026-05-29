import matplotlib.pyplot as plt 
import json 
import omegaconf
import numpy as np
from hydra.core.hydra_config import HydraConfig
from pathlib import Path



def ema_smooth(values, alpha=0.1):
    """
    Exponential moving average smoothing
    Args:
        values: list or 1D numpy array
        alpha: smoothing factor in (0,1)
        returns smoothed numpy array
    """
    values = np.array(values, dtype=float)
    if len(values) == 0:
        return values

    ema = np.zeros_like(values)
    ema[0] = values[0]
    for i in range(1, len(values)):
        ema[i] = alpha * values[i] + (1 - alpha) * ema[i-1]
    return ema


def plot_losses(config : omegaconf.DictConfig):

    training_history_json = config.history_json

    with open(training_history_json, "r", encoding = "utf-8" ) as f: 
         training_history = json.load(f)

    epochs_record  = training_history["epochs"]
    alpha = config.alpha

    epochs = []
    train_losses = []
    val_losses = []
    point_errors = []
    epoch_times = []

    for item in epochs_record : 
         epochs.append(item["epoch"])
         train_losses.append(item["train_loss"])
         val_losses.append(item["val_loss"])
         point_errors.append(item["val_mean_point_error"])
         epoch_times.append(item["epoch_time"])
    

    epochs = np.asarray(epochs)
    train_losses = np.asarray(train_losses)
    val_losses = np.asarray(val_losses)
    point_errors = np.asarray(point_errors)
    epoch_times = np.asarray(epoch_times)

    # smooth for visualization
    train_losses = ema_smooth(train_losses, alpha)
    val_losses = ema_smooth(val_losses, alpha)
    point_errors = ema_smooth(point_errors, alpha)


    plt.figure()
    plt.plot(epochs, train_losses, color="green", label="Train loss")
    plt.plot(epochs, val_losses, color="blue", label="Validation loss")
   

    plt.xlabel("Epoch")
    plt.legend()
    plt.grid(True)

    out_path = Path(HydraConfig.get().runtime.output_dir) / "losses.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.show()

    plt.figure()
    plt.plot(epochs, epoch_times, color="black", label="Epoch time")
    plt.xlabel("Epoch")
    plt.legend()
    plt.grid(True)

    out_path_times = Path(HydraConfig.get().runtime.output_dir) / "epoch_times.png"
    plt.savefig(out_path_times, dpi=300, bbox_inches="tight")

    plt.figure()
    plt.plot(epochs, point_errors, color="red", label="Euclidean mean point error")
    plt.xlabel("Epoch")
    plt.legend()
    plt.grid(True)
    out_path_errors = Path(HydraConfig.get().runtime.output_dir) / "errors.png" 
    plt.savefig(out_path_errors, dpi = 300, bbox_inches = "tight")

         

