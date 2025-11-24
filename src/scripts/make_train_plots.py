import os

from aquarel import load_theme
import dotenv
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from tqdm import tqdm
import wandb

from utils.deep_evaluation_analysis_utils import format

THEME = load_theme("scientific").set_font(sans_serif="DejaVu Sans").set_transforms(trim=True)
SAVE_PATH = os.path.join(os.getcwd(), "reports", "figures", "training_plots")


def get_training_runs():
    """Fetch training metrics from the run from WanDB."""

    api = wandb.Api()
    runs = api.runs(
        path=f"{os.environ.get('WANDB_ENTITY')}/{os.environ.get('WANDB_PROJECT')}",
        filters={"state": "finished", "jobType": "training", "tags": {"$ne": "OLD"}},
    )

    return runs


def extract_run_metadata_from_job_name(run_name: str):
    job_name = "_".join(run_name.split("_")[0:-1])

    model = job_name.split("_")[0].split("-")[0]

    spec_augment = "spec-aug" in job_name
    pitch_shift = "pitch-shift" in job_name
    speed_perturb = "speed-perturbations" in job_name

    return {
        "job_name": job_name,
        "run_id": run_name.split("_")[-1],
        "model": model,
        "spec_augment": spec_augment,
        "pitch_shift": pitch_shift,
        "speed_perturb": speed_perturb,
    }


def process_run(run) -> dict:
    run_name = run.name
    history = run.scan_history()

    run_metadata = extract_run_metadata_from_job_name(run_name)
    train_loss = []
    val_wer = []
    learning_rates = []

    for sample in history:
        # Collect training loss

        if "train_loss" in sample and sample["train_loss"] is not None:
            train_loss.append((sample["trainer/global_step"], sample["train_loss"]))

        if (
            "coralval_wer" in sample
            and sample["coralval_wer"] is not None
            and "fleursval_wer" in sample
            and sample["fleursval_wer"] is not None
        ):
            val_wer.append(
                (sample["trainer/global_step"], sample["coralval_wer"], sample["fleursval_wer"])
            )

        if "learning_rate" in sample and sample["learning_rate"] is not None:
            learning_rates.append((sample["trainer/global_step"], sample["learning_rate"]))

    run_data = {
        "run_metadata": run_metadata,
        "train_loss": train_loss,
        "val_wer": val_wer,
        "learning_rates": learning_rates,
    }

    return run_data


def add_augmentations_legend(
    ax, spec_augment: bool, pitch_shift: bool, speed_perturb: bool, regular_legend: bool = False
):
    # 1. Handle the Regular Legend First
    if regular_legend:
        # Create the standard legend for lines already plotted on the axes
        # (e.g., "Training Loss")
        first_legend = ax.legend(
            frameon=True,
            fontsize=10,
            loc="upper right",
            borderaxespad=0.1,
        )

        # CRITICAL STEP: Add this legend as a separate artist so it isn't
        # overwritten by the second legend call below.
        ax.add_artist(first_legend)

    # Add legend with augmentation status
    legend_handles = []

    legend_handles.append(mpatches.Patch(color="none", label="$\\bf{Augmentations}$"))

    COLOR_ACTIVATED = "#2ecc71"
    COLOR_DEACTIVATED = "#e60505"

    STATUS_TEXT_ACTIVATED = "ON"
    STATUS_TEXT_DEACTIVATED = "OFF"

    # Spec Augment
    patch = mpatches.Patch(
        color=COLOR_ACTIVATED if spec_augment else COLOR_DEACTIVATED,
        label=f"Spec Augment: {STATUS_TEXT_ACTIVATED if spec_augment else STATUS_TEXT_DEACTIVATED}",
    )
    legend_handles.append(patch)

    # Pitch Shift
    patch = mpatches.Patch(
        color=COLOR_ACTIVATED if pitch_shift else COLOR_DEACTIVATED,
        label=f"Pitch Shift: {STATUS_TEXT_ACTIVATED if pitch_shift else STATUS_TEXT_DEACTIVATED}",
    )
    legend_handles.append(patch)

    # Speed Perturbations
    patch = mpatches.Patch(
        color=COLOR_ACTIVATED if speed_perturb else COLOR_DEACTIVATED,
        label=f"Speed Perturbations: {STATUS_TEXT_ACTIVATED if speed_perturb else STATUS_TEXT_DEACTIVATED}",
    )
    legend_handles.append(patch)

    ax.legend(
        handles=legend_handles,
        bbox_to_anchor=(0.47, -0.15),
        loc="upper center",
        ncol=len(legend_handles),
        frameon=True,
        fontsize=10,
    )

    # Add a regular legend frame if specified


def plot_train_loss(run_data: dict):
    with THEME:
        fig, ax = plt.subplots(figsize=(10, 4))

        steps, losses = zip(*run_data["train_loss"])

        # Plot training loss
        ax.plot(steps, losses, linewidth=1.5, alpha=0.9, label="Train Loss")

        run_md = run_data["run_metadata"]

        add_augmentations_legend(
            ax, run_md["spec_augment"], run_md["pitch_shift"], run_md["speed_perturb"]
        )

        ax.set_title(f"Train Loss for {format(run_data['run_metadata']['model'])}")
        ax.set_xlabel("Steps")
        ax.set_ylabel("Loss")

        plt.tight_layout()

    os.makedirs(SAVE_PATH, exist_ok=True)
    plt.savefig(
        os.path.join(SAVE_PATH, f"train_loss_{run_data['run_metadata']['job_name']}.png"), dpi=300
    )


def plot_individual_val_wer(run_data: dict):
    with THEME:
        fig, ax = plt.subplots(figsize=(10, 4))

        steps, coral_wers, fleurs_wers = zip(*run_data["val_wer"])

        # Plot validation WERs
        ax.plot(
            steps,
            coral_wers,
            linewidth=1.5,
            alpha=0.9,
            label=f"{format('coral')} Validation {format('WER')}",
        )
        ax.plot(
            steps,
            fleurs_wers,
            linewidth=1.5,
            alpha=0.9,
            label=f"{format('fleurs')} Validation {format('WER')}",
        )

        run_md = run_data["run_metadata"]

        add_augmentations_legend(
            ax,
            run_md["spec_augment"],
            run_md["pitch_shift"],
            run_md["speed_perturb"],
            regular_legend=True,
        )

        ax.set_title(f"Validation {format('WER')} for {format(run_data['run_metadata']['model'])}")
        ax.set_xlabel("Steps")
        ax.set_ylabel("WER")
        # ax.legend()

        plt.tight_layout()

    os.makedirs(SAVE_PATH, exist_ok=True)
    plt.savefig(
        os.path.join(SAVE_PATH, f"val_wer_{run_data['run_metadata']['job_name']}.png"), dpi=300
    )


def plot_val_wer(processed_runs: list[dict]):
    def _plot(model: str, dataset: str):
        with THEME:
            fig, ax = plt.subplots(figsize=(10, 4))

            for run_data in processed_runs:
                if run_data["run_metadata"]["model"] != model:
                    continue

                steps, coral_wers, fleurs_wers = zip(*run_data["val_wer"])

                augs = []

                if run_data["run_metadata"]["spec_augment"]:
                    augs.append("SA")
                if run_data["run_metadata"]["pitch_shift"]:
                    augs.append("PS")
                if run_data["run_metadata"]["speed_perturb"]:
                    augs.append("SP")

                label = "+".join(augs) if augs else "No Augmentations"

                # Plot validation WERs
                ax.plot(
                    steps,
                    coral_wers if dataset == "coral" else fleurs_wers,
                    linewidth=1.5,
                    alpha=0.5,
                    label=label,
                )

            ax.set_title(f"{format(dataset)} Validation {format('WER')} for {format(model)}")
            ax.set_xlabel("Steps")
            ax.set_ylabel("WER")
            ax.legend()

            plt.tight_layout()

            os.makedirs(SAVE_PATH, exist_ok=True)
            plt.savefig(os.path.join(SAVE_PATH, f"val_wer_{model}_{dataset}.png"), dpi=300)

    _plot("canary", "coral")
    _plot("canary", "fleurs")
    _plot("parakeet", "coral")
    _plot("parakeet", "fleurs")


if __name__ == "__main__":
    dotenv.load_dotenv()

    runs = get_training_runs()
    processed_runs = [process_run(run) for run in tqdm(runs)]

    # Plot training losses
    for run_data in processed_runs:
        plot_train_loss(run_data)

    # Plot individual validation WERs
    for run_data in processed_runs:
        plot_individual_val_wer(run_data)

    # Plot combined validation WERs
    plot_val_wer(processed_runs)
