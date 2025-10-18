from __future__ import annotations

import os

from dotenv import load_dotenv
from subjob import Submittor
from subjob.lsf import LSFSubmissionOptions
from subjob.lsf.options import GPUMode

if __name__ == "__main__":
    load_dotenv()
    job_name = "train-model"

    opts = LSFSubmissionOptions(
        queue="gpua100",
        job_name=job_name,
        num_cores=8,
        gpu_mode=GPUMode.EXCLUSIVE_PROCESS,
        gpu_num=1,
        walltime="01:00",
        memory="2GB",
        working_directory=os.environ.get("HPC_PATH"),
        # Uncomment to direct outputs:
        output_file=f"logs/{job_name}.%J.out",
        error_file=f"logs/{job_name}.%J.err",
    )

    with Submittor(opts) as s:
        s.sync_packages_uv()
        s.activate_venv(".venv")
        s.command(
            [
                "python",
                "src/scripts/train_model.py",
            ]
        )
