from __future__ import annotations

import os

from dotenv import load_dotenv
from subjob import Submittor
from subjob.lsf import LSFSubmissionOptions


if __name__ == "__main__":
    load_dotenv()
    job_name = "download-nemo-dataset"

    opts = LSFSubmissionOptions(
        queue="hpc",
        job_name=job_name,
        num_cores=32,
        walltime="24:00",
        memory="2GB",
        working_directory=os.environ.get("HPC_PATH"),
        # Uncomment to direct outputs:
        output_file=f"logs/{job_name}.%J.out",
        error_file=f"logs/{job_name}.%J.err",
    )

    with Submittor(opts) as s:
        s.sync_packages_uv()
        s.activate_venv(".venv")
        s.command(["just", "nemo-dataset-prepare"])
