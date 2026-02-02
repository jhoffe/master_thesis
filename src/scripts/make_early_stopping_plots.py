import os
from pathlib import Path

import dotenv
import matplotlib.pyplot as plt
import pandas as pd
import typer

SAVE_PATH = os.path.join(os.getcwd(), "reports", "figures", "early_stopping_plots")

app = typer.Typer()


def pick_best_after_patience(val_wer_series: pd.Series, min_epochs: int, patience: int) -> int:
    best_epoch = val_wer_series[min_epochs:].idxmin()
    best_wer = val_wer_series[best_epoch]

    for epoch in range(min_epochs, len(val_wer_series)):
        if val_wer_series[epoch] < best_wer:
            best_wer = val_wer_series[epoch]
            best_epoch = epoch
        elif epoch - best_epoch >= patience:
            break

    return best_epoch


@app.command()
def main(
    val_csv_path: Path = typer.Argument(
        ..., help="Path to the CSV file containing validation data."
    ),
    train_loss_csv_path: Path = typer.Argument(
        ..., help="Path to the CSV file containing training loss data."
    ),
    min_epochs: int = typer.Option(
        5, help="Minimum number of epochs before early stopping can occur."
    ),
    patience: int = typer.Option(3, help="Patience for early stopping."),
):
    """Generate training plots from WandB runs."""
    dotenv.load_dotenv()

    val_df = pd.read_csv(val_csv_path)
    train_loss_df = pd.read_csv(train_loss_csv_path)

    val_wer_x = val_df["trainer/global_step"]
    val_wer_y = val_df["val_wer"]

    min_epochs_line_x = val_wer_x[min_epochs - 1]
    best_epoch = pick_best_after_patience(val_wer_y, min_epochs, patience)

    train_loss_x = train_loss_df["trainer/global_step"]
    train_loss_y = train_loss_df["train_loss"]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax2 = ax.twinx()

    line1 = ax.plot(
        train_loss_x,
        train_loss_y,
        color="tab:blue",
        label="Training Loss",
    )
    min_epochs_line = ax.axvline(
        x=min_epochs_line_x,
        color="tab:orange",
        linestyle=":",
        label="Min. Epochs",
    )
    best_epoch_line = ax.axvline(
        x=val_wer_x[best_epoch],
        color="tab:green",
        linestyle="--",
        label="Best Epoch",
    )
    line2 = ax2.plot(
        val_wer_x,
        val_wer_y,
        color="tab:red",
        label="Validation WER",
    )

    ax.set_xlabel("Steps")
    ax.set_ylabel("Training Loss")
    ax2.set_ylabel("Validation WER")

    # Combine legends
    lines = line1 + line2 + [min_epochs_line, best_epoch_line]
    labels = [l.get_label() for l in lines]
    ax.legend(lines, labels, loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=2, frameon=True)

    plt.tight_layout()
    os.makedirs(SAVE_PATH, exist_ok=True)
    plot_save_path = os.path.join(SAVE_PATH, "early_stopping_plot.png")
    plt.savefig(plot_save_path, dpi=300)
    print(f"Plot saved to {plot_save_path}")


if __name__ == "__main__":
    app()
