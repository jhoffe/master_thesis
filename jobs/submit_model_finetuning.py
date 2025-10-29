from __future__ import annotations

import os

from dotenv import load_dotenv
from subjob import Submittor
from subjob.lsf import LSFSubmissionOptions
from subjob.lsf.options import GPUMode


def submit_job(config_name: str):
    job_name = f"finetune-model-{config_name}"

    opts = LSFSubmissionOptions(
        queue="gpua100",
        job_name=job_name,
        num_cores=12,
        gpu_mode=GPUMode.EXCLUSIVE_PROCESS,
        gpu_num=1,
        walltime="32:00",
        memory="5GB",
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


if __name__ == "__main__":
    load_dotenv()

    submit_job("canary-1b-v2-finetune")
    submit_job("parakeet-tdt-0.6b-v3-finetune")
