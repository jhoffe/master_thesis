from __future__ import annotations

from enum import Enum
from math import e
import os
from typing import Annotated, Literal

from dotenv import load_dotenv
from subjob import Submittor
from subjob.lsf import LSFSubmissionOptions


def main():
    job_name = "kenlm-parakeet"

    opts = LSFSubmissionOptions(
        queue="hpc",
        job_name=job_name,
        num_cores=16,
        walltime="16:00",
        memory="16GB",
        working_directory=os.environ.get("HPC_PATH"),
        # Uncomment to direct outputs:
        output_file=f"logs/{job_name}.%J.out",
        error_file=f"logs/{job_name}.%J.err",
        email=os.environ.get("HPC_EMAIL_ADDRESS"),
        notify_on_start=os.environ.get("HPC_NOTIFY_ON_START") == "1",
        notify_on_completion=os.environ.get("HPC_NOTIFY_ON_COMPLETION") == "1",
    )

    with Submittor(opts, verbosity=2) as s:
        s.sync_packages_uv()
        s.activate_venv(".venv")

        s.command(["just", "train-kenlm"])


if __name__ == "__main__":
    load_dotenv()
    main()
