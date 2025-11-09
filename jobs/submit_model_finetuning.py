from __future__ import annotations

import os

from dotenv import load_dotenv
from subjob import Submittor
from subjob.lsf import LSFSubmissionOptions
from subjob.lsf.options import GPUMode
import typer


def submit_job(config_name: str):
    job_name = f"finetune-model-{config_name}"

    opts = LSFSubmissionOptions(
        queue="gpua100",
        job_name=job_name,
        num_cores=9,
        gpu_mode=GPUMode.EXCLUSIVE_PROCESS,
        gpu_num=1,
        walltime="24:00",
        memory="7GB",
        working_directory=os.environ.get("HPC_PATH"),
        # Uncomment to direct outputs:
        output_file=f"logs/{job_name}.%J.out",
        error_file=f"logs/{job_name}.%J.err",
        environment={
            "NUMBA_CUDA_USE_NVIDIA_BINDING": "0",
            "WANDB_JOB_TYPE": "training",
        },
        additional_resources=["select[gpu80gb]"],
        email=os.environ.get("HPC_EMAIL_ADDRESS"),
        notify_on_start=os.environ.get("HPC_NOTIFY_ON_START") == "1",
        notify_on_completion=os.environ.get("HPC_NOTIFY_ON_COMPLETION") == "1",
    )

    with Submittor(opts) as s:
        s.sync_packages_uv()
        s.activate_venv(".venv")

        s.command(
            [
                "python",
                "src/scripts/train_model.py",
                "--config-name",
                f"{config_name}.yaml",
            ]
        )


app = typer.Typer()


@app.command()
def main(
    config_names: list[str] = typer.Argument(
        None,
        help="List of config names to submit. If not provided, submits all default configs.",
    ),
):
    """
    Submit model finetuning jobs to the cluster.

    Examples:
        # Submit all default configs
        python jobs/submit_model_finetuning.py

        # Submit specific configs
        python jobs/submit_model_finetuning.py parakeet-tdt-0.6b-v3-finetune canary-1b-v2-finetune
    """
    load_dotenv()

    # Default configs to submit if none specified
    default_configs = [
        # "parakeet-tdt-0.6b-v3-finetune_spec-aug",
        # "canary-1b-v2-finetune_spec-aug",
        # "parakeet-tdt-0.6b-v3-finetune",
        # "canary-1b-v2-finetune",
        # "canary-1b-v2-finetune_12-buckets",
        "canary-1b-v2-finetune_speech_perturbations_30-buckets",
        "canary-1b-v2-finetune_spec-aug_speech_perturbations_30-buckets",
        "parakeet-tdt-0.6b-v3-finetune_speech_perturbations",
        "parakeet-tdt-0.6b-v3-finetune_spec-aug_speech_perturbations",
        "parakeet-tdt-0.6b-v3-finetune",
    ]

    # Use provided configs or default to all
    configs_to_submit = config_names if config_names else default_configs

    typer.echo(f"Submitting {len(configs_to_submit)} job(s)...")
    for config_name in configs_to_submit:
        typer.echo(f"  - Submitting: {config_name}")
        submit_job(config_name)

    typer.echo("✓ All jobs submitted successfully!")


if __name__ == "__main__":
    app()
