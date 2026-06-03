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

    #training_history_json = Path('/home/renku/work/s3-bucket/OUTPUTS/unet_coord_runs/2026-06-01_11-53-33/training_history.json')
    training_history_json = config.history_json
    with open(training_history_json, "r", encoding = "utf-8" ) as f: 
         training_history = json.load(f)

    epochs_record  = training_history["epochs"]
    alpha_1 = config.alpha_1
    alpha_2 = config.alpha_2

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
    train_losses = ema_smooth(train_losses, alpha_1)
    val_losses = ema_smooth(val_losses, alpha_1)
    point_errors = ema_smooth(point_errors, alpha_2)


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




def plot_losses_test(results: dict,
    output_subdir: str = "inference_metric_plots",
):
    """
    Load inference_outputs.json and plot, for each sample key:
        - loss
        - mean_point_error_normalized
        - mean_point_error_voxel
    """
    output_dir = Path(HydraConfig.get().runtime.output_dir)

    samples = results.get("samples", [])

    if len(samples) == 0:
        raise RuntimeError("No samples found in %s. Skipping metric plots.", json_path)

    samples = sorted(
        samples,
        key=lambda s: (
            s.get("batch_id", 0),
            s.get("sample_index_in_batch", 0),
        ),
    )

    keys = [str(s["key"]) for s in samples]

    metrics = {
        "loss": [s["loss"] for s in samples],
        "mean_point_error_normalized": [
            s["mean_point_error_normalized"] for s in samples
        ],
        "mean_point_error_voxel": [
            s["mean_point_error_voxel"] for s in samples
        ],
    }

    plot_dir = output_dir / output_subdir
    plot_dir.mkdir(parents=True, exist_ok=True)

    x = np.arange(len(keys[:50]))

    # avoid unreadable x-axis if many samples
    max_labels = 10
    label_step = max(1, len(keys[:50]) // max_labels)

    for metric_name, values in metrics.items():
        plt.figure(figsize=(max(10, len(keys[:50]) * 0.35), 5))

        plt.plot(x[:50], values[:50], marker="o", linewidth=1)

        plt.xticks(
            x[::label_step],
            [(keys[:50])[i] for i in x[::label_step]],
            rotation=90,)

        plt.xlabel("Test frames")
        plt.ylabel(metric_name)
        plt.title(f"{metric_name} per test sample")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        out_file = plot_dir / f"{metric_name}_per_key.png"
        plt.savefig(out_file, dpi=200)
        plt.close()

        print(f"Saved {metric_name} plot to: {out_file}")

         

