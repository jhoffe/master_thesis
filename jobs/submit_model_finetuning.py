import os
from time import sleep

from dotenv import load_dotenv
from subjob import Submittor
from subjob.lsf import LSFSubmissionOptions
from subjob.lsf.options import GPUMode


def submit_job(config_name: str, walltime: str, wait_for: str | None = None):
    job_name = f"finetune-model-{config_name}"

    opts = LSFSubmissionOptions(
        queue="gpuh100",
        job_name=job_name,
        num_cores=24,
        gpu_mode=GPUMode.EXCLUSIVE_PROCESS,
        gpu_num=1,
        walltime=walltime,
        memory="4GB",
        working_directory=os.environ.get("HPC_PATH"),
        dependency=wait_for,
        # Uncomment to direct outputs:
        output_file=f"logs/{job_name}.%J.out",
        error_file=f"logs/{job_name}.%J.err",
        environment={
            "NUMBA_CUDA_USE_NVIDIA_BINDING": "0",
            "WANDB_JOB_TYPE": "training",
        },
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

    return s.get_job_id()


def main():
    """
    Submit model finetuning jobs to the cluster.
    """
    load_dotenv()

    experiments = [
        "",
        "_pitch-shift",
        "_spec-aug_pitch-shift",
        "_spec-aug_speed-perturbations_pitch-shift",
        "_spec-aug_speed-perturbations",
        "_spec-aug",
        "_speed-perturbations_pitch-shift",
        "_speed-perturbations",
    ]

    # Default configs to submit if none specified
    canary_configs = [f"canary-finetune{exp}" for exp in experiments]
    parakeet_configs = [f"parakeet-finetune{exp}" for exp in experiments]

    job_id = None

    for config_name in canary_configs:
        job_id = submit_job(config_name, walltime="20:00", wait_for=job_id)
        sleep(2)  # Slight delay to avoid overwhelming the scheduler

    job_id = None

    for config_name in parakeet_configs:
        job_id = submit_job(config_name, walltime="17:00", wait_for=job_id)
        sleep(2)  # Slight delay to avoid overwhelming the scheduler


if __name__ == "__main__":
    main()
