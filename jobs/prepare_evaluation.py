from __future__ import annotations

import os

from dotenv import load_dotenv
from subjob import Submittor
from subjob.lsf import LSFSubmissionOptions


if __name__ == "__main__":
    load_dotenv()
    job_name = "prepare-evaluation"

    opts = LSFSubmissionOptions(
        queue="hpc",
        job_name=job_name,
        num_cores=32,
        walltime="1:00",
        memory="2GB",
        working_directory=os.environ.get("HPC_PATH"),
        # Uncomment to direct outputs:
        output_file=f"logs/{job_name}.%J.out",
        error_file=f"logs/{job_name}.%J.err",
        email=os.environ.get("HPC_EMAIL_ADDRESS"),
        notify_on_start=os.environ.get("HPC_NOTIFY_ON_START") == "1",
        notify_on_completion=os.environ.get("HPC_NOTIFY_ON_COMPLETION") == "1",
    )

    with Submittor(opts) as s:
        #s.sync_packages_uv()
        s.activate_venv(".venv")
        s.command(["python", "src/scripts/prepare_evaluation.py"])
