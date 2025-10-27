from __future__ import annotations

from enum import Enum
from math import e
import os
from typing import Annotated, Literal

from dotenv import load_dotenv
from subjob import Submittor
from subjob.lsf import LSFSubmissionOptions
from subjob.lsf.options import GPUMode
import typer


class GPUMem(str, Enum):
    GB40 = "40gb"
    GB80 = "80gb"


def main(gpu_mem: Annotated[GPUMem, typer.Option()] = GPUMem.GB80):
    job_name = f"oomptimizer-gpu{gpu_mem}"

    opts = LSFSubmissionOptions(
        queue="gpua100",
        job_name=job_name,
        num_cores=4,
        gpu_mode=GPUMode.EXCLUSIVE_PROCESS,
        gpu_num=1,
        walltime="00:30",
        memory="8GB",
        working_directory=os.environ.get("HPC_PATH"),
        # Uncomment to direct outputs:
        output_file=f"logs/{job_name}.%J.out",
        error_file=f"logs/{job_name}.%J.err",
        environment={
            "NUMBA_CUDA_USE_NVIDIA_BINDING": "0",
            "WANDB_JOB_TYPE": "training",
        },
        additional_resources=[f"select[gpu{gpu_mem}]"],
        email=os.environ.get("HPC_EMAIL_ADDRESS"),
        notify_on_start=os.environ.get("HPC_NOTIFY_ON_START") == "1",
        notify_on_completion=os.environ.get("HPC_NOTIFY_ON_COMPLETION") == "1",
    )

    with Submittor(opts, verbosity=2) as s:
        s.sync_packages_uv()
        s.activate_venv(".venv")

        s.command(["just", "nemo-oomptimize"])


if __name__ == "__main__":
    load_dotenv()
    typer.run(main)
